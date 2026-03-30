import ReactMarkdown from 'react-markdown';
import { User, Bot, ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

function CitationList({ citations }) {
  const [open, setOpen] = useState(false);
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {citations.length} source{citations.length > 1 ? 's' : ''}
      </button>
      {open && (
        <div className="mt-2 space-y-1.5">
          {citations.map((c, i) => (
            <div key={i} className="text-xs text-gray-500 bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">
              <span className="font-medium text-gray-600">[{i + 1}]</span>{' '}
              {[c.paper_id, c.section, c.page && `Page ${c.page}`].filter(Boolean).join(' | ') || 'Unknown source'}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ role, content, citations }) {
  const isUser = role === 'user';

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center shrink-0 mt-1">
          <Bot size={15} className="text-gray-500" />
        </div>
      )}
      <div className={`max-w-[75%] ${isUser ? 'order-first' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-black text-white'
              : 'bg-gray-50 text-gray-800 border border-gray-100'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{content}</p>
          ) : (
            <div className="prose prose-sm prose-gray max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && <CitationList citations={citations} />}
      </div>
      {isUser && (
        <div className="w-7 h-7 rounded-full bg-black flex items-center justify-center shrink-0 mt-1">
          <User size={15} className="text-white" />
        </div>
      )}
    </div>
  );
}

export default function ChatMessages({ messages, loading }) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
        {messages.length === 0 && !loading && (
          <div className="text-center text-gray-400 mt-32">
            <Bot size={40} className="mx-auto mb-4 text-gray-300" />
            <p className="text-lg font-medium text-gray-500">Scholar RAG</p>
            <p className="text-sm mt-1">Upload papers and ask questions</p>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} citations={m.citations} />
        ))}
        {loading && !messages.some((m) => m.role === 'assistant' && !m.finalized) && (
          <div className="flex gap-4">
            <div className="w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center shrink-0">
              <Bot size={15} className="text-gray-500" />
            </div>
            <div className="bg-gray-50 rounded-2xl px-4 py-3 border border-gray-100">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
