import React, { useEffect, useMemo, useRef, useState } from "react";
import ChatThread from "./components/ChatThread.jsx";
import LiteratureMap from "./components/LiteratureMap.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { addArxiv, deletePaper, getPapers, streamChat, uploadPdf } from "./utils/api.js";

const STORAGE_KEY = "research-chat-conversations";

const loadConversations = () => {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
};

const newConversation = () => ({
  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  title: "",
  messages: [],
});

function App() {
  const [conversations, setConversations] = useState(loadConversations);
  const [activeId, setActiveId] = useState(() => conversations[0]?.id ?? null);
  const [streaming, setStreaming] = useState(false);
  const [papers, setPapers] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState("");
  const [view, setView] = useState("chat");
  const [dragging, setDragging] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch {
      /* storage full or unavailable — chat still works, history just won't persist */
    }
  }, [conversations]);

  useEffect(() => {
    getPapers().then(setPapers).catch(() => {});
  }, []);

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId]
  );

  const patchConversation = (id, patch) =>
    setConversations((prev) => prev.map((c) => (c.id === id ? patch(c) : c)));

  const patchLastMessage = (id, patch) =>
    patchConversation(id, (c) => ({
      ...c,
      messages: c.messages.map((m, i) => (i === c.messages.length - 1 ? patch(m) : m)),
    }));

  const stopStreaming = () => abortRef.current?.abort();

  // `base` overrides the conversation's messages (used by regenerate to
  // truncate the thread before resending).
  const sendMessage = async (text, base = null) => {
    abortRef.current?.abort();
    let convo = active;
    if (!convo) {
      convo = newConversation();
      setConversations((prev) => [convo, ...prev]);
      setActiveId(convo.id);
    }
    const id = convo.id;
    const baseMessages = base ?? convo.messages;
    const history = baseMessages
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.content || "" }));

    patchConversation(id, (c) => ({
      ...c,
      title: c.title || text.slice(0, 60),
      messages: [
        ...baseMessages,
        { role: "user", content: text },
        { role: "assistant", content: "", streaming: true, stage: "retrieving" },
      ],
    }));

    const controller = new AbortController();
    abortRef.current = controller;
    // A stream that dies after tokens keeps the partial text instead of
    // replacing it with the error.
    const interrupt = (note) =>
      patchLastMessage(id, (m) =>
        m.content
          ? { ...m, streaming: false, interrupted: true, note }
          : { ...m, streaming: false, error: note }
      );

    setStreaming(true);
    try {
      await streamChat(
        [...history, { role: "user", content: text }],
        (event) => {
          if (event.type === "token") {
            patchLastMessage(id, (m) => ({ ...m, content: m.content + event.text }));
          } else if (event.type === "stage") {
            patchLastMessage(id, (m) => ({ ...m, stage: event.name }));
          } else if (event.type === "done") {
            patchLastMessage(id, () => ({
              role: "assistant",
              content: event.answer || "",
              matches: event.matches || [],
              claims: event.claims || [],
              verification: event.verification || null,
              stances: event.stances || {},
              warnings: event.warnings || [],
              suggestions: event.suggestions || [],
            }));
          } else if (event.type === "error") {
            interrupt(event.detail);
          }
        },
        controller.signal
      );
    } catch (err) {
      interrupt(
        controller.signal.aborted
          ? "Stopped."
          : err.message || "That question could not be answered."
      );
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setStreaming(false);
    }
  };

  const regenerate = () => {
    if (!active || streaming) return;
    const msgs = active.messages;
    let i = msgs.length - 1;
    while (i >= 0 && msgs[i].role !== "user") i--;
    if (i < 0) return;
    sendMessage(msgs[i].content, msgs.slice(0, i));
  };

  const selectConversation = (id) => {
    abortRef.current?.abort();
    setActiveId(id);
  };

  const addPaper = async (fetchPaper) => {
    try {
      setUploading(true);
      setUploadStatus("Reading, chunking and embedding — this takes a minute.");
      setError("");
      const res = await fetchPaper();
      setUploadStatus(`Added ${res.filename}: ${res.newChunks} passages.`);
      setPapers(await getPapers());
      return true;
    } catch (err) {
      setUploadStatus("");
      setError(err.message || "Could not add that paper.");
      return false;
    } finally {
      setUploading(false);
    }
  };

  const addFiles = async (fileList) => {
    const pdfs = Array.from(fileList).filter(
      (f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name)
    );
    if (!pdfs.length) return false;
    setUploading(true);
    setUploadStatus("");
    setError("");
    setJobs(pdfs.map((f) => ({ name: f.name, status: "queued", detail: "" })));
    const setJob = (i, patch) =>
      setJobs((prev) => prev.map((j, k) => (k === i ? { ...j, ...patch } : j)));
    // ponytail: sequential uploads — server ingest is synchronous behind a
    // global lock, so parallel POSTs would only queue there anyway.
    for (let i = 0; i < pdfs.length; i++) {
      setJob(i, { status: "ingesting" });
      try {
        const res = await uploadPdf(pdfs[i]);
        setJob(i, { status: "done", detail: `${res.newChunks} passages` });
      } catch (err) {
        setJob(i, { status: "failed", detail: err.message || "failed" });
      }
    }
    getPapers().then(setPapers).catch(() => {});
    setUploading(false);
    return true;
  };

  const removePaper = async (file) => {
    try {
      setError("");
      await deletePaper(file);
      setPapers(await getPapers());
    } catch (err) {
      setError(err.message || "Could not remove that paper.");
    }
  };

  const deleteConversation = (id) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (id === activeId) setActiveId(null);
  };

  const renameConversation = (id, title) =>
    patchConversation(id, (c) => ({ ...c, title: title.trim() || c.title }));

  return (
    <div
      className={`app${dragging ? " dragging" : ""}`}
      onDragOver={(e) => {
        if (e.dataTransfer?.types?.includes("Files")) {
          e.preventDefault();
          setDragging(true);
        }
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer?.files?.length && !uploading) addFiles(e.dataTransfer.files);
      }}
    >
      <header className="masthead">
        <div className="wordmark">
          <span className="mark">Research</span> Assistant
        </div>
        <nav className="viewtoggle">
          <button
            type="button"
            className={view === "chat" ? "active" : ""}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
          <button
            type="button"
            className={view === "map" ? "active" : ""}
            onClick={() => setView("map")}
          >
            Map
          </button>
        </nav>
        <dl className="stats">
          <div>
            <dt>Papers</dt>
            <dd>{papers.length || "—"}</dd>
          </div>
        </dl>
      </header>

      <main className="layout">
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={selectConversation}
          onNew={() => selectConversation(null)}
          onDelete={deleteConversation}
          onRename={renameConversation}
          papers={papers}
          onDeletePaper={removePaper}
          onUploadFiles={(files) => (files?.length ? addFiles(files) : false)}
          onAddArxiv={(url) => addPaper(() => addArxiv(url))}
          uploading={uploading}
          uploadStatus={uploadStatus}
          jobs={jobs}
          error={error}
        />
        {view === "chat" ? (
          <ChatThread
            messages={active?.messages ?? []}
            streaming={streaming}
            onSend={sendMessage}
            onStop={stopStreaming}
            onRegenerate={regenerate}
            papers={papers}
          />
        ) : (
          <LiteratureMap paperCount={papers.length} />
        )}
      </main>
      {dragging && <div className="dropzone-hint">Drop PDFs to add them to the corpus</div>}
    </div>
  );
}

export default App;
