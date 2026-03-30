"""File upload and management APIs."""

import uuid
import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles

from config import Config
from app.dependencies import get_pdf_parser, get_rag_integration, get_retriever
from app.store import add_file, list_files, get_file, delete_file_record, get_file_by_hash

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    upload_dir = Path(Config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    results = []

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            results.append({"filename": f.filename, "status": "error", "detail": "Only PDF files are supported"})
            continue

        content = await f.read()
        if len(content) > max_bytes:
            results.append({"filename": f.filename, "status": "error", "detail": f"File exceeds {Config.MAX_UPLOAD_SIZE_MB}MB limit"})
            continue

        content_hash = hashlib.sha256(content).hexdigest()
        existing = get_file_by_hash(content_hash)
        if existing:
            results.append({"filename": f.filename, "status": "duplicate", "detail": f"Same content as '{existing['filename']}'"})
            continue

        file_id = str(uuid.uuid4())
        paper_id = Path(f.filename).stem
        save_path = upload_dir / f"{file_id}.pdf"

        async with aiofiles.open(save_path, "wb") as out:
            await out.write(content)

        try:
            parser = get_pdf_parser()
            nodes = parser.parse(str(save_path), paper_id)

            integration = get_rag_integration()
            docs = integration.nodes_to_documents(nodes)
            parents, children = integration.create_chunks(docs)
            integration.store_in_milvus(parents, children)

            record = add_file(
                file_id=file_id,
                filename=f.filename,
                paper_id=paper_id,
                content_hash=content_hash,
                size_bytes=len(content),
                page_count=max((n.page_num for n in nodes), default=0),
                chunk_count=len(children),
            )
            results.append({"filename": f.filename, "status": "ok", **record})
        except Exception as e:
            logger.exception(f"Failed to process {f.filename}")
            save_path.unlink(missing_ok=True)
            results.append({"filename": f.filename, "status": "error", "detail": str(e)})

    return {"files": results}


@router.get("")
async def get_files():
    return list_files()


@router.delete("/{file_id}")
async def remove_file(file_id: str):
    record = delete_file_record(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    retriever = get_retriever()
    updater = retriever.get_updater()
    updater.delete_paper(record["paper_id"])

    save_path = Path(Config.UPLOAD_DIR) / f"{file_id}.pdf"
    save_path.unlink(missing_ok=True)

    return {"ok": True, "paper_id": record["paper_id"]}
