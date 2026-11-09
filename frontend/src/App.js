// import React, { useState, useRef, useEffect } from 'react';
// import './style.css';

// const API_URL = (
//   import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
// ).replace(/\/$/, '');

// const createSessionId = () => {
//   if (typeof crypto !== 'undefined' && crypto.randomUUID) {
//     return crypto.randomUUID();
//   }

//   return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
// };

// const getErrorMessage = (data, fallback) => {
//   if (typeof data?.detail === 'string') {
//     return data.detail;
//   }

//   if (Array.isArray(data?.detail)) {
//     return data.detail
//       .map(item => item.msg || JSON.stringify(item))
//       .join(', ');
//   }

//   return data?.message || data?.error || fallback;
// };

// const normalizeCitation = (citation, index) => ({
//   id: citation?.id || citation?.evidence_id || citation?.citation_id || `source-${index + 1}`,
//   source:
//     citation?.file_name ||
//     citation?.source ||
//     citation?.filename ||
//     citation?.source_path ||
//     'Unknown source',
//   fileType: citation?.file_type || '',
//   page: citation?.page ?? citation?.page_number ?? null,
//   chunkId: citation?.chunk_id || null,
//   content: citation?.content || citation?.text || '',
//   cited: citation?.cited,
// });

// const App = () => {
//   const [messages, setMessages] = useState([]);
//   const [inputText, setInputText] = useState('');
//   const [isLoading, setIsLoading] = useState(false);

//   const [sessionId, setSessionId] = useState(createSessionId);

//   const [isIngesting, setIsIngesting] = useState(false);
//   const [ingestionStatus, setIngestionStatus] = useState(null);

//   const chatEndRef = useRef(null);
//   const inputRef = useRef(null);

//   useEffect(() => {
//     chatEndRef.current?.scrollIntoView({
//       behavior: 'smooth',
//     });
//   }, [messages, isLoading]);

//   const handleSend = async () => {
//     const question = inputText.trim();

//     if (!question || isLoading) return;

//     const userMessage = {
//       id: `${Date.now()}-user`,
//       sender: 'user',
//       text: question,
//     };

//     setMessages(prev => [...prev, userMessage]);
//     setInputText('');
//     setIsLoading(true);

//     try {
//       const response = await fetch(`${API_URL}/ask`, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           question,
//           session_id: sessionId,
//         }),
//       });

//       const data = await response.json().catch(() => ({}));

//       if (!response.ok) {
//         throw new Error(
//           getErrorMessage(data, `Request failed (${response.status})`)
//         );
//       }

//       const citations = Array.isArray(data.citations)
//         ? data.citations.map(normalizeCitation)
//         : [];

//       const aiMessage = {
//         id: `${Date.now()}-ai`,
//         sender: 'ai',
//         text: data.answer || 'No response received from the backend.',
//         citations,
//         retrieval: data.retrieval || null,
//       };

//       setMessages(prev => [...prev, aiMessage]);

//     } catch (error) {
//       setMessages(prev => [
//         ...prev,
//         {
//           id: `${Date.now()}-error`,
//           sender: 'ai',
//           text: `⚠️ ${error.message || 'Backend not reachable.'}`,
//           citations: [],
//           isError: true,
//         },
//       ]);
//     } finally {
//       setIsLoading(false);
//       inputRef.current?.focus();
//     }
//   };

//   const handleNewChat = () => {
//     setMessages([]);
//     setInputText('');
//     setSessionId(createSessionId());
//     setIngestionStatus(null);
//     inputRef.current?.focus();
//   };

//   const handleIngest = async () => {
//     if (isIngesting || isLoading) return;

//     setIsIngesting(true);
//     setIngestionStatus({
//       type: 'loading',
//       message: 'Processing documents and building the knowledge base...',
//     });

//     try {
//       const response = await fetch(`${API_URL}/ingest`, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//       });

//       const data = await response.json().catch(() => ({}));

//       if (!response.ok) {
//         throw new Error(
//           getErrorMessage(data, `Ingestion failed (${response.status})`)
//         );
//       }

//       const report = data.report || data;

//       setIngestionStatus({
//         type: 'success',
//         message: 'Knowledge base updated successfully.',
//         report,
//       });

//     } catch (error) {
//       setIngestionStatus({
//         type: 'error',
//         message: error.message || 'Document ingestion failed.',
//       });
//     } finally {
//       setIsIngesting(false);
//     }
//   };

//   const renderCitations = citations => {
//     if (!citations || citations.length === 0) return null;

//     return (
//       <div className="citations">
//         <div className="citations-heading">
//           📚 Sources
//         </div>

//         {citations.map((citation, index) => (
//           <div className="citation-item" key={`${citation.id}-${index}`}>
//             <div className="citation-title">
//               [{citation.id}] {citation.source}
//             </div>

//             <div className="citation-details">
//               {citation.fileType && (
//                 <span>{citation.fileType.toUpperCase()}</span>
//               )}

//               {citation.page !== null && citation.page !== undefined && (
//                 <span>Page {citation.page}</span>
//               )}

//               {citation.chunkId && (
//                 <span>Chunk: {citation.chunkId.slice(0, 12)}</span>
//               )}
//             </div>

//             {citation.cited === false && (
//               <span className="citation-status">
//                 Retrieved evidence — not explicitly cited
//               </span>
//             )}

//             {citation.content && (
//               <details className="citation-content">
//                 <summary>View supporting content</summary>
//                 <p>{citation.content}</p>
//               </details>
//             )}
//           </div>
//         ))}
//       </div>
//     );
//   };

//   const renderRetrievalInfo = retrieval => {
//     if (!retrieval) return null;

//     const retrieved =
//       retrieval.retrieved_count ??
//       retrieval.candidate_count;

//     const reranked =
//       retrieval.reranked_count ??
//       retrieval.selected_count;

//     if (retrieved === undefined && reranked === undefined) {
//       return null;
//     }

//     return (
//       <div className="retrieval-info">
//         {retrieved !== undefined && (
//           <span>Retrieved: {retrieved}</span>
//         )}

//         {reranked !== undefined && (
//           <span>Selected: {reranked}</span>
//         )}
//       </div>
//     );
//   };

//   return (
//     <div className="app-wrapper">

//       {/* Ambient background */}
//       <div className="ambient-glow glow-top"></div>
//       <div className="ambient-glow glow-bottom"></div>

//       {/* Sidebar */}
//       <div className="sidebar">

//         <h2 className="logo">CortexRAG</h2>

//         <button
//           className="new-chat"
//           onClick={handleNewChat}
//           disabled={isLoading}
//         >
//           + New Chat
//         </button>

//         <button
//           className="new-chat ingest-btn"
//           onClick={handleIngest}
//           disabled={isIngesting || isLoading}
//         >
//           {isIngesting ? '⏳ Processing...' : '↻ Update Knowledge Base'}
//         </button>

//         {ingestionStatus && (
//           <div className={`ingestion-status ${ingestionStatus.type}`}>
//             <p>{ingestionStatus.message}</p>

//             {ingestionStatus.report && (
//               <div className="ingestion-report">
//                 <p>
//                   Discovered: {ingestionStatus.report.discovered ?? '-'}
//                 </p>

//                 <p>
//                   Successful: {
//                     ingestionStatus.report.successful?.length ?? '-'
//                   }
//                 </p>

//                 <p>
//                   Failed: {
//                     ingestionStatus.report.failed?.length ?? '-'
//                   }
//                 </p>

//                 <p>
//                   Total chunks: {
//                     ingestionStatus.report.total_chunks ?? '-'
//                   }
//                 </p>
//               </div>
//             )}
//           </div>
//         )}

//         <div className="features">
//           <div className="feature">📄 Multi-Document RAG</div>
//           <div className="feature">⚡ Hybrid Retrieval</div>
//           <div className="feature">🎯 Reciprocal Rank Fusion</div>
//           <div className="feature">🧠 Cross-Encoder Reranking</div>
//           <div className="feature">📚 Evidence-Based Citations</div>
//           <div className="feature">💬 Conversational Memory</div>
//         </div>

//       </div>

//       {/* Main */}
//       <div className="main">

//         <div className="content">

//           {messages.length === 0 ? (

//             <div className="empty">

//               <div className="orb">
//                 <span>✨</span>
//               </div>

//               <h1>Ask anything</h1>

//               <p>
//                 Your AI knowledge assistant powered by hybrid RAG
//               </p>

//             </div>

//           ) : (

//             <div className="chat">

//               {messages.map(msg => (

//                 <div
//                   key={msg.id}
//                   className={`msg ${msg.sender}`}
//                 >

//                   <div className="message-text">
//                     {msg.text}
//                   </div>

//                   {msg.sender === 'ai' && !msg.isError && (
//                     <>
//                       {renderCitations(msg.citations)}
//                       {renderRetrievalInfo(msg.retrieval)}
//                     </>
//                   )}

//                 </div>

//               ))}

//               {isLoading && (
//                 <div className="msg ai typing">
//                   <span></span>
//                   <span></span>
//                   <span></span>
//                 </div>
//               )}

//               <div ref={chatEndRef} />

//             </div>

//           )}

//         </div>

//         {/* Input */}
//         <div className="input-wrapper">

//           <div className="input-bar">

//             <input
//               ref={inputRef}
//               type="text"
//               placeholder="Ask anything..."
//               value={inputText}
//               disabled={isLoading}
//               onChange={e => setInputText(e.target.value)}
//               onKeyDown={e => {
//                 if (e.key === 'Enter' && !e.shiftKey) {
//                   e.preventDefault();
//                   handleSend();
//                 }
//               }}
//             />

//             <button
//               onClick={handleSend}
//               className="send-btn"
//               disabled={isLoading || !inputText.trim()}
//               aria-label="Send message"
//             >
//               ✨
//             </button>

//           </div>

//         </div>

//       </div>

//     </div>
//   );
// };

// export default App;




import React, { useState, useRef, useEffect } from 'react';
import './style.css';

const API_URL = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/$/, '');

const createSessionId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

const getErrorMessage = (data, fallback) => {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail)) {
    return data.detail.map(item => item.msg || JSON.stringify(item)).join(', ');
  }
  return data?.message || data?.error || fallback;
};

const normalizeCitation = (citation, index) => ({
  id: citation?.id || citation?.evidence_id || citation?.citation_id || `source-${index + 1}`,
  source:
    citation?.file_name ||
    citation?.source ||
    citation?.filename ||
    citation?.source_path ||
    'Unknown source',
  fileType: citation?.file_type || '',
  page: citation?.page ?? citation?.page_number ?? null,
  chunkId: citation?.chunk_id || null,
  content: citation?.content || citation?.text || '',
  cited: citation?.cited,
});

const normalizeDocumentSource = source => ({
  status: source?.status || 'unknown',
  answer: source?.answer || 'No document answer returned.',
  citations: Array.isArray(source?.citations)
    ? source.citations.map(normalizeCitation)
    : [],
  retrieval: source?.retrieval || null,
  error: source?.error || source?.message || null,
});

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [mode, setMode] = useState('document');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(createSessionId);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionStatus, setIngestionStatus] = useState(null);

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    const question = inputText.trim();
    if (!question || isLoading) return;

    const requestedMode = mode;
    const userMessage = {
      id: `${Date.now()}-user`,
      sender: 'user',
      text: question,
      mode: requestedMode,
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          session_id: sessionId,
          mode: requestedMode,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(getErrorMessage(data, `Request failed (${response.status})`));
      }

      let aiMessage;

      if (requestedMode === 'both') {
        const documentSource = normalizeDocumentSource(data.sources?.document);
        const databaseSource = data.sources?.database || {
          status: 'unknown',
          error: 'No database source information returned.',
        };

        aiMessage = {
          id: `${Date.now()}-ai`,
          sender: 'ai',
          mode: requestedMode,
          text: data.answer || 'No combined response received.',
          documentSource,
          databaseSource,
        };
      } else if (requestedMode === 'database') {
        aiMessage = {
          id: `${Date.now()}-ai`,
          sender: 'ai',
          mode: requestedMode,
          text: data.answer || 'No database response received.',
          databaseSource: data.database || null,
        };
      } else {
        aiMessage = {
          id: `${Date.now()}-ai`,
          sender: 'ai',
          mode: requestedMode,
          text: data.answer || 'No response received from the backend.',
          citations: Array.isArray(data.citations)
            ? data.citations.map(normalizeCitation)
            : [],
          retrieval: data.retrieval || null,
        };
      }

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          id: `${Date.now()}-error`,
          sender: 'ai',
          text: `⚠️ ${error.message || 'Backend not reachable.'}`,
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setInputText('');
    setSessionId(createSessionId());
    setIngestionStatus(null);
    inputRef.current?.focus();
  };

  const handleIngest = async () => {
    if (isIngesting || isLoading) return;

    setIsIngesting(true);
    setIngestionStatus({
      type: 'loading',
      message: 'Processing documents and building the knowledge base...',
    });

    try {
      const response = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(getErrorMessage(data, `Ingestion failed (${response.status})`));
      }

      const report = data.report || data;
      setIngestionStatus({
        type: 'success',
        message: 'Knowledge base updated successfully.',
        report,
      });
    } catch (error) {
      setIngestionStatus({
        type: 'error',
        message: error.message || 'Document ingestion failed.',
      });
    } finally {
      setIsIngesting(false);
    }
  };

  const renderCitations = citations => {
    if (!citations?.length) return null;

    return (
      <div className="citations">
        <div className="citations-heading">📚 Document sources</div>
        {citations.map((citation, index) => (
          <div className="citation-item" key={`${citation.id}-${index}`}>
            <div className="citation-title">
              [{citation.id}] {citation.source}
            </div>
            <div className="citation-details">
              {citation.fileType && <span>{citation.fileType.toUpperCase()}</span>}
              {citation.page !== null && citation.page !== undefined && (
                <span>Page {citation.page}</span>
              )}
              {citation.chunkId && <span>Chunk: {citation.chunkId.slice(0, 12)}</span>}
            </div>
            {citation.cited === false && (
              <span className="citation-status">
                Retrieved evidence — not explicitly cited
              </span>
            )}
            {citation.content && (
              <details className="citation-content">
                <summary>View supporting content</summary>
                <p>{citation.content}</p>
              </details>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderRetrievalInfo = retrieval => {
    if (!retrieval) return null;

    const retrieved = retrieval.retrieved_count ?? retrieval.candidate_count ??
      retrieval.retrieved_evidence_count;
    const selected = retrieval.reranked_count ?? retrieval.selected_count;

    if (retrieved === undefined && selected === undefined) return null;

    return (
      <div className="retrieval-info">
        {retrieved !== undefined && <span>Retrieved: {retrieved}</span>}
        {selected !== undefined && <span>Selected: {selected}</span>}
        {retrieval.cited_evidence_count !== undefined && (
          <span>Cited: {retrieval.cited_evidence_count}</span>
        )}
      </div>
    );
  };

  const renderDatabaseSource = database => {
    if (!database) return null;

    const evidence = database.evidence;
    const metadata = database.metadata;

    return (
      <section className="source-panel database-panel">
        <div className="source-panel-heading">
          <span>🗄️ Database results</span>
          <span className={`source-status ${database.status || 'unknown'}`}>
            {database.status || 'unknown'}
          </span>
        </div>

        {database.error && <p className="source-error">{database.error}</p>}
        {database.answer && <p className="source-answer">{database.answer}</p>}

        {evidence && (
          <>
            {evidence.source_views?.length > 0 && (
              <div className="database-meta">
                <strong>Source views:</strong> {evidence.source_views.join(', ')}
              </div>
            )}
            {evidence.sql && (
              <details className="sql-details">
                <summary>View generated SQL</summary>
                <pre>{evidence.sql}</pre>
              </details>
            )}
            {Array.isArray(evidence.columns) && Array.isArray(evidence.rows) && (
              <div className="table-scroll">
                <table className="results-table">
                  <thead>
                    <tr>
                      {evidence.columns.map(column => <th key={column}>{column}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {evidence.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {evidence.columns.map(column => (
                          <td key={column}>{String(row?.[column] ?? '')}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="database-meta">
              Rows: {evidence.row_count ?? evidence.rows?.length ?? 0}
              {evidence.truncated ? ' · Results truncated' : ''}
              {evidence.row_limit ? ` · Limit: ${evidence.row_limit}` : ''}
            </div>
          </>
        )}

        {metadata && (
          <div className="retrieval-info">
            {metadata.strategy && <span>Strategy: {metadata.strategy}</span>}
            {metadata.intent && <span>Intent: {metadata.intent}</span>}
            {metadata.elapsed_ms !== undefined && <span>{metadata.elapsed_ms} ms</span>}
          </div>
        )}
        {metadata?.warnings?.length > 0 && (
          <ul className="source-warnings">
            {metadata.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
          </ul>
        )}
      </section>
    );
  };

  const renderDocumentSource = document => {
    if (!document) return null;

    return (
      <section className="source-panel document-panel">
        <div className="source-panel-heading">
          <span>📄 Document RAG</span>
          <span className={`source-status ${document.status || 'unknown'}`}>
            {document.status || 'unknown'}
          </span>
        </div>
        {document.error && <p className="source-error">{document.error}</p>}
        <p className="source-answer">{document.answer}</p>
        {renderCitations(document.citations)}
        {renderRetrievalInfo(document.retrieval)}
      </section>
    );
  };

  return (
    <div className="app-wrapper">
      <div className="ambient-glow glow-top"></div>
      <div className="ambient-glow glow-bottom"></div>

      <aside className="sidebar">
        <h2 className="logo">CortexRAG</h2>

        <button className="new-chat" onClick={handleNewChat} disabled={isLoading}>
          + New Chat
        </button>

        <button
          className="new-chat ingest-btn"
          onClick={handleIngest}
          disabled={isIngesting || isLoading}
        >
          {isIngesting ? '⏳ Processing...' : '↻ Update Knowledge Base'}
        </button>

        {ingestionStatus && (
          <div className={`ingestion-status ${ingestionStatus.type}`}>
            <p>{ingestionStatus.message}</p>
            {ingestionStatus.report && (
              <div className="ingestion-report">
                <p>Discovered: {ingestionStatus.report.discovered ?? '-'}</p>
                <p>Successful: {ingestionStatus.report.successful?.length ?? '-'}</p>
                <p>Failed: {ingestionStatus.report.failed?.length ?? '-'}</p>
                <p>Total chunks: {ingestionStatus.report.total_chunks ?? '-'}</p>
              </div>
            )}
          </div>
        )}

        <div className="features">
          <div className="feature">📄 Multi-Document RAG</div>
          <div className="feature">⚡ Hybrid Retrieval</div>
          <div className="feature">🎯 Reciprocal Rank Fusion</div>
          <div className="feature">🧠 Cross-Encoder Reranking</div>
          <div className="feature">📚 Evidence-Based Citations</div>
          <div className="feature">💬 Conversational Memory</div>
        </div>
      </aside>

      <main className="main">
        <div className="content">
          {messages.length === 0 ? (
            <div className="empty">
              <div className="orb"><span>✨</span></div>
              <h1>Ask anything</h1>
              <p>Your AI knowledge assistant powered by hybrid RAG</p>
            </div>
          ) : (
            <div className="chat">
              {messages.map(msg => (
                <div key={msg.id} className={`msg ${msg.sender}`}>
                  {msg.sender === 'user' && msg.mode && (
                    <div className="message-mode">{msg.mode} mode</div>
                  )}
                  <div className="message-text">{msg.text}</div>

                  {msg.sender === 'ai' && !msg.isError && (
                    <>
                      {msg.mode === 'both' ? (
                        <div className="combined-sources">
                          {renderDocumentSource(msg.documentSource)}
                          {renderDatabaseSource(msg.databaseSource)}
                        </div>
                      ) : msg.mode === 'database' ? (
                        renderDatabaseSource(msg.databaseSource)
                      ) : (
                        <>
                          {renderCitations(msg.citations)}
                          {renderRetrievalInfo(msg.retrieval)}
                        </>
                      )}
                    </>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="msg ai typing">
                  <span></span><span></span><span></span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        <div className="input-wrapper">
          <div className="input-composer">
            <div className="mode-control">
              <label htmlFor="query-mode">Query mode</label>
              <select
                id="query-mode"
                value={mode}
                onChange={event => setMode(event.target.value)}
                disabled={isLoading}
              >
                <option value="document">Document RAG</option>
                <option value="database">Database SQL</option>
                <option value="both">Both sources</option>
              </select>
            </div>

            <div className="input-bar">
              <input
                ref={inputRef}
                type="text"
                placeholder={
                  mode === 'document'
                    ? 'Ask about your documents...'
                    : mode === 'database'
                      ? 'Ask a question about the Olist database...'
                      : 'Ask across documents and the database...'
                }
                value={inputText}
                disabled={isLoading}
                onChange={event => setInputText(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    handleSend();
                  }
                }}
              />
              <button
                onClick={handleSend}
                className="send-btn"
                disabled={isLoading || !inputText.trim()}
                aria-label="Send message"
              >
                ✨
              </button>
            </div>
            <p className="mode-hint">
              {mode === 'document' && 'Searches your ingested documents.'}
              {mode === 'database' && 'Queries the connected Olist database.'}
              {mode === 'both' && 'Runs both sources independently and keeps their evidence separate.'}
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
