import { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Settings, ChevronLeft } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatMessages from './components/ChatMessages';
import ChatInput from './components/ChatInput';
import FileUpload from './components/FileUpload';
import SettingsPanel from './components/SettingsPanel';
import { fetchSessions, fetchHistory, deleteSession, streamChat } from './api';

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [panel, setPanel] = useState(null);
  const cancelRef = useRef(null);
  const bottomRef = useRef(null);

  const loadSessions = useCallback(() => {
    fetchSessions().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => { loadSessions(); }, [loadSessions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const selectSession = async (id) => {
    setCurrentId(id);
    setPanel(null);
    try {
      const data = await fetchHistory(id);
      setMessages(data.messages || []);
    } catch {
      setMessages([]);
    }
  };

  const newChat = () => {
    setCurrentId(null);
    setMessages([]);
    setPanel(null);
  };

  const handleDelete = async (id) => {
    await deleteSession(id);
    if (currentId === id) newChat();
    loadSessions();
  };

  const handleSend = (query) => {
    if (cancelRef.current) cancelRef.current();

    setMessages((prev) => [...prev, { role: 'user', content: query }]);
    setLoading(true);

    let sessionId = currentId;
    let answer = '';
    let citations = [];

    const cancel = streamChat(query, sessionId, (evt) => {
      switch (evt.type) {
        case 'session_id':
          sessionId = evt.data;
          setCurrentId(evt.data);
          break;
        case 'answer':
          answer = evt.data;
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === 'assistant' && !last.finalized) {
              return prev.slice(0, -1).concat({ ...last, content: answer });
            }
            return [...prev, { role: 'assistant', content: answer, citations: [], finalized: false }];
          });
          break;
        case 'citations':
          citations = evt.data;
          setMessages((prev) =>
            prev.map((m, i) =>
              i === prev.length - 1 && m.role === 'assistant'
                ? { ...m, citations, finalized: true }
                : m
            )
          );
          break;
        case 'done':
          setLoading(false);
          loadSessions();
          break;
        case 'error':
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: `Error: ${evt.data}`, citations: [] },
          ]);
          setLoading(false);
          break;
      }
    });

    cancelRef.current = cancel;
  };

  const togglePanel = (name) => setPanel((prev) => (prev === name ? null : name));

  return (
    <div className="flex h-screen bg-white">
      <Sidebar
        sessions={sessions}
        currentId={currentId}
        onSelect={selectSession}
        onNew={newChat}
        onDelete={handleDelete}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 bg-white">
          <h1 className="text-sm font-medium text-gray-700">Scholar RAG</h1>
          <div className="flex items-center gap-1">
            <button
              onClick={() => togglePanel('upload')}
              className={`p-2 rounded-lg transition-colors ${panel === 'upload' ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
            >
              <Upload size={16} className="text-gray-500" />
            </button>
            <button
              onClick={() => togglePanel('settings')}
              className={`p-2 rounded-lg transition-colors ${panel === 'settings' ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
            >
              <Settings size={16} className="text-gray-500" />
            </button>
          </div>
        </header>

        <div className="flex flex-1 min-h-0">
          <div className="flex-1 flex flex-col min-w-0">
            <ChatMessages messages={messages} loading={loading} />
            <div ref={bottomRef} />
            <ChatInput onSend={handleSend} disabled={loading} />
          </div>

          {panel && (
            <div className="w-72 border-l border-gray-200 bg-white overflow-y-auto">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-100">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {panel === 'upload' ? 'Upload Files' : 'Settings'}
                </span>
                <button onClick={() => setPanel(null)} className="p-1 rounded hover:bg-gray-100">
                  <ChevronLeft size={14} className="text-gray-400" />
                </button>
              </div>
              {panel === 'upload' && <FileUpload onUploaded={loadSessions} />}
              {panel === 'settings' && <SettingsPanel onCollectionCleared={loadSessions} />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
