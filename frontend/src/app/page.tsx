"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("fraud liability");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [copied, setCopied] = useState(false);

  const translate = async () => {
    if (!query) return;
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error("API Error");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to API. Is the FastAPI backend running?");
    } finally {
      setLoading(false);
    }
  };

  const copyResult = async () => {
    if (!result) return;
    const textToCopy = JSON.stringify(result, null, 2);
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy", err);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex items-center justify-center p-4 font-sans">
      <div className="max-w-3xl w-full bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden flex flex-col min-h-[500px]">
        {/* Header */}
        <div className="relative bg-indigo-900/95 backdrop-blur-md text-white px-8 py-6 flex items-center justify-between shadow-md z-10 overflow-hidden">
          <div className="absolute -right-8 -top-8 w-32 h-32 bg-indigo-500 rounded-full opacity-20 blur-2xl"></div>
          <div className="absolute -left-8 -bottom-8 w-24 h-24 bg-blue-400 rounded-full opacity-20 blur-2xl"></div>
          
          <div className="relative z-10">
            <h1 className="text-2xl font-bold tracking-tight">Legal RAG</h1>
            <div className="flex items-center gap-2 mt-1">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
              <p className="text-indigo-200 text-sm font-medium">Query Translation Engine</p>
            </div>
          </div>
          
          <div className="relative z-10 w-12 h-12 bg-white/10 rounded-full flex items-center justify-center border border-white/20 backdrop-blur-sm shadow-inner group">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-50 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-8 bg-slate-50 relative rounded-b-2xl">
          <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#312e81 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
          
          {/* Section 1: Input */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 relative group transition-all duration-300 hover:shadow-md hover:border-indigo-200 z-10">
            <label htmlFor="queryInput" className="block text-sm font-bold text-slate-700 mb-2 uppercase tracking-wider">Analyze Legal Query</label>
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
                  </svg>
                </div>
                <input 
                  type="text" 
                  id="queryInput" 
                  placeholder="e.g. fraud liability" 
                  className="w-full pl-11 pr-4 py-3.5 border border-slate-300 rounded-lg focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all outline-none text-slate-700 bg-slate-50 focus:bg-white text-base shadow-sm"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && translate()}
                />
              </div>
              <button 
                onClick={translate}
                disabled={loading}
                className="px-8 py-3.5 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 disabled:opacity-75 disabled:cursor-wait text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2 active:scale-[0.98]">
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Translating...
                  </>
                ) : (
                  <>
                    <span>Translate</span>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Section 2: Output */}
          {result && (
            <div className="flex flex-col gap-5 z-10 transition-all opacity-100 animate-[fadeIn_0.5s_ease-out_forwards]">
              <style dangerouslySetInnerHTML={{__html: `
                @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
              `}} />
              <div className="flex justify-between items-center px-1">
                <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                  Transformation Steps
                </h2>
                <div className="flex gap-2 items-center">
                  {copied && (
                    <span className="text-sm text-green-600 font-bold px-4 py-2 bg-green-50 border border-green-200 rounded-lg shadow-sm flex items-center gap-1 animate-pulse">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      Copied!
                    </span>
                  )}
                  {!copied && (
                    <button onClick={copyResult} className="text-sm px-4 py-2 flex items-center gap-2 text-slate-600 hover:text-indigo-700 hover:bg-indigo-50 bg-white border border-slate-200 rounded-lg shadow-sm transition-all focus:ring-2 focus:ring-indigo-100 outline-none">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy JSON
                    </button>
                  )}
                </div>
              </div>

              {/* Original */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
                <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-slate-400 group-hover:bg-slate-500 transition-colors"></div>
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                  Original Input
                </h3>
                <p className="text-slate-800 font-mono text-sm bg-white p-3 rounded-lg border border-slate-100">
                  "{result.original}"
                </p>
              </div>

              {/* Expanded */}
              <div className="bg-blue-50/70 border border-blue-100 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
                <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-blue-500 group-hover:bg-blue-600 transition-colors"></div>
                <h3 className="text-xs font-bold text-blue-700 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <div className="p-1 bg-blue-100 rounded-full">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                    </svg>
                  </div>
                  Expanded Query
                </h3>
                <p className="text-blue-900 font-medium bg-white/60 p-3 rounded-lg border border-blue-100/50 leading-relaxed">
                  {result.expanded}
                </p>
              </div>

              {/* Rewritten */}
              <div className="bg-green-50/70 border border-green-100 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
                <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-green-500 group-hover:bg-green-600 transition-colors"></div>
                <h3 className="text-xs font-bold text-green-700 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <div className="p-1 bg-green-100 rounded-full">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </div>
                  Rewritten Form
                </h3>
                <p className="text-green-900 font-medium bg-white/60 p-3 rounded-lg border border-green-100/50 text-lg">
                  {result.rewritten}
                </p>
              </div>

              {/* Decomposed */}
              <div className="bg-purple-50/70 border border-purple-100 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
                <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-purple-500 group-hover:bg-purple-600 transition-colors"></div>
                <h3 className="text-xs font-bold text-purple-700 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <div className="p-1 bg-purple-100 rounded-full">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6z" />
                    </svg>
                  </div>
                  Query Decomposition
                </h3>
                <ul className="flex flex-col gap-2 mt-1">
                  {result.decomposed.map((item: string, index: number) => (
                    <li key={index} className="bg-white p-3 rounded-lg border border-purple-100/60 shadow-sm flex items-start gap-3 group/item hover:border-purple-300 transition-colors">
                      <div className="bg-purple-100 text-purple-700 font-bold text-xs rounded-full w-5 h-5 flex items-center justify-center shrink-0 mt-0.5 group-hover/item:bg-purple-600 group-hover/item:text-white transition-colors">
                        {index + 1}
                      </div>
                      <span className="text-purple-900 font-medium">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
