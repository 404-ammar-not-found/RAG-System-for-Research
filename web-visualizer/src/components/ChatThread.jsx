import React, { useEffect, useMemo, useRef, useState } from "react";
import Message from "./Message.jsx";

function ChatThread({ messages, streaming, onSend, onStop, onRegenerate, papers = [] }) {
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const starters = useMemo(() => {
    const title = papers[Math.floor(Math.random() * papers.length)]?.title;
    return [
      title && `What problem does “${title}” solve?`,
      papers.length > 1 && "Where do these papers disagree?",
      "Summarize the methods used across these papers",
    ].filter(Boolean);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [papers]);

  const send = (text) => {
    if (!text || streaming) return;
    setInput("");
    onSend(text);
  };

  return (
    <div className="thread-wrap">
      <div className="thread" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty hero">
            <p>
              Ask a question about the papers in your library. Every claim comes back
              with a numbered citation you can open.
            </p>
            {papers.length > 0 && (
              <div className="chips">
                {starters.map((s) => (
                  <button key={s} type="button" className="chip" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {messages.map((m, i) => (
          <Message
            key={i}
            message={m}
            isLast={i === messages.length - 1}
            streaming={streaming}
            onSend={send}
            onRegenerate={onRegenerate}
          />
        ))}
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input.trim());
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input.trim());
            }
            if (e.key === "Escape" && streaming) onStop();
          }}
          placeholder="Which papers use attention, and what did they evaluate on?"
          rows={2}
        />
        {streaming ? (
          <button className="primary stop" type="button" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="primary" type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}

export default ChatThread;
