"""Test PDF parser with real PDF documents."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.integration import PDFParser


def test_parse_pdf(pdf_path: str, save_log: bool = True):
    """Test parsing a PDF file."""
    parser = PDFParser()
    
    output_lines = []
    
    def log(msg):
        print(msg)
        output_lines.append(msg)
    
    log(f"Parsing: {pdf_path}")
    log("=" * 80)
    
    try:
        nodes = parser.parse(pdf_path, "test_paper")
        
        log(f"\n✓ Successfully parsed {len(nodes)} nodes\n")
        
        # Statistics
        type_counts = {}
        for node in nodes:
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
        
        log("Node Type Distribution:")
        log("-" * 40)
        for node_type, count in sorted(type_counts.items()):
            log(f"  {node_type:20s}: {count:4d}")
        
        # Show samples by type
        log("\n" + "=" * 80)
        log("Sample Nodes by Type:")
        log("=" * 80)
        
        for node_type in sorted(type_counts.keys()):
            samples = [n for n in nodes if n.node_type == node_type][:2]
            if samples:
                log(f"\n[{node_type.upper()}]")
                log("-" * 80)
                for i, node in enumerate(samples, 1):
                    log(f"\nSample {i}:")
                    log(f"  Page: {node.page_num}, Order: {node.order}")
                    if node.section_path:
                        log(f"  Section: {' > '.join(node.section_path)}")
                    if node.bbox:
                        log(f"  BBox: ({node.bbox[0]:.1f}, {node.bbox[1]:.1f}, {node.bbox[2]:.1f}, {node.bbox[3]:.1f})")
                    log(f"  Text ({len(node.text)} chars):")
                    preview = node.text[:200].replace('\n', ' ')
                    log(f"    {preview}...")
        
        # Full node dump
        log("\n" + "=" * 80)
        log("ALL NODES (Full Details):")
        log("=" * 80)
        
        for node in nodes:
            log(f"\n{'='*80}")
            log(f"Node #{node.order} | Type: {node.node_type} | Page: {node.page_num}")
            log(f"{'='*80}")
            if node.section_path:
                log(f"Section: {' > '.join(node.section_path)}")
            if node.bbox:
                log(f"BBox: {node.bbox}")
            log(f"\nText:\n{node.text}\n")
        
        # Save to log file
        if save_log:
            log_dir = Path(__file__).parent.parent / "log"
            log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = Path(pdf_path).stem
            log_file = log_dir / f"parse_{pdf_name}_{timestamp}.log"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            
            print(f"\n✓ Log saved to: {log_file}")
        
        return nodes
        
    except Exception as e:
        log(f"\n✗ Error: {e}")
        import traceback
        log(traceback.format_exc())
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_parser.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_parse_pdf(pdf_path)



