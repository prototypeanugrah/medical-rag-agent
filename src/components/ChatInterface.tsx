'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '@/types';
import { getApiUrl, API_ENDPOINTS } from '@/lib/api-config';
import { useTheme } from '@/contexts/ThemeContext';
import ThemeToggle from '@/components/ThemeToggle';

interface AgentResponse {
  content: string;
  reasoning: string[];
  warnings: Array<{
    type: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
  }>;
  sources: string[];
  confidence: number;
  followUpQuestions?: string[];
}

interface ChatInterfaceProps {
  sessionId?: string;
}

const ThinkingAnimation = () => (
  <div className="flex items-center space-x-2 p-4">
    <div className="flex space-x-1">
      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
    </div>
    <span className="text-gray-400 text-sm">AI is thinking...</span>
  </div>
);

const WelcomeHeader = ({ isCompact }: { isCompact: boolean }) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  
  return (
    <div className={`text-center transition-all duration-500 ${isCompact ? 'py-4' : 'py-8'} animate-fade-in relative`}>
      {isCompact && (
        <div className="absolute top-4 right-0">
          <ThemeToggle />
        </div>
      )}
      <div className={`bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto shadow-2xl transition-all duration-500 ${
        isCompact ? 'w-12 h-12 mb-3' : 'w-20 h-20 mb-6'
      }`}>
        <svg className={`text-white transition-all duration-500 ${isCompact ? 'w-6 h-6' : 'w-10 h-10'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      </div>
      <h1 className={`font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent transition-all duration-500 ${
        isCompact ? 'text-2xl mb-2' : 'text-4xl mb-4'
      }`}>
        Medical RAG Assistant
      </h1>
      {!isCompact && (
        <>
          <p className={`text-xl max-w-2xl mx-auto leading-relaxed transition-opacity duration-500 ${
            isDark ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Get reliable, evidence-based information about drug interactions, contraindications, and medical safety from verified medical databases.
          </p>
          <div className="mt-6">
            <ThemeToggle />
          </div>
        </>
      )}
    </div>
  );
};

const ExampleCards = ({ onExampleClick }: { onExampleClick: (text: string) => void }) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  
  const examples = [
    {
      icon: "🩺",
      text: "I'm currently taking Lepirudin and my doctor just prescribed Apixaban. Are there any interactions I should be worried about?",
      category: "Drug Interactions"
    },
    {
      icon: "🍎",
      text: "I'm on Warfarin for blood clots. What foods should I avoid?",
      category: "Dietary Guidelines"
    },
    {
      icon: "💊",
      text: "I have high blood pressure and take Lisinopril. Can I safely add ibuprofen for joint pain?",
      category: "Medication Safety"
    },
    {
      icon: "⚠️",
      text: "What are the contraindications for taking aspirin with my current medications?",
      category: "Contraindications"
    }
  ];

  return (
    <div className="flex flex-col items-center justify-center px-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl w-full mb-12">
        {examples.map((example, index) => (
          <div
            key={index}
            onClick={() => onExampleClick(example.text)}
            className={`group rounded-xl p-6 cursor-pointer hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 ${
              isDark 
                ? 'bg-gray-800 border border-gray-700 hover:border-blue-500' 
                : 'bg-white border border-gray-200 hover:border-blue-300'
            }`}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-start space-x-4">
              <div className="text-2xl flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                {example.icon}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-blue-500 uppercase tracking-wide mb-2">
                  {example.category}
                </div>
                <p className={`text-sm leading-relaxed transition-colors duration-300 ${
                  isDark 
                    ? 'text-gray-300 group-hover:text-white' 
                    : 'text-gray-700 group-hover:text-gray-900'
                }`}>
                  "{example.text}"
                </p>
              </div>
              <svg className={`w-5 h-5 transition-colors duration-300 ${
                isDark 
                  ? 'text-gray-500 group-hover:text-blue-400' 
                  : 'text-gray-400 group-hover:text-blue-500'
              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </div>
          </div>
        ))}
      </div>

      <div className="text-center">
        <p className={`text-sm mb-4 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
          Powered by advanced AI and medical databases
        </p>
        <div className={`flex items-center justify-center space-x-6 text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span>Evidence-based</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span>Verified sources</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
            <span>Real-time analysis</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const MessageBubble = ({ message, onFollowUpClick }: { message: ChatMessage; onFollowUpClick: (text: string) => void }) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const isUser = message.role === 'user';
  
  if (isUser) {
    return (
      <div className="flex justify-end mb-6 animate-slide-in-right">
        <div className="max-w-3xl bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-2xl rounded-tr-md px-6 py-4 shadow-lg">
          <p className="leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start mb-8 animate-slide-in-left">
      <div className="max-w-4xl w-full">
        <div className="flex items-start space-x-3 mb-3">
          <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-md">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div className={`flex-1 rounded-2xl rounded-tl-md px-6 py-5 shadow-sm ${
            isDark 
              ? 'bg-gray-800 border border-gray-700' 
              : 'bg-white border border-gray-200'
          }`}>
            
            {/* Warnings Section */}
            {message.metadata?.warnings && message.metadata.warnings.length > 0 && (
              <div className="mb-6 space-y-3">
                {message.metadata.warnings.map((warning: any, idx: number) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-xl border-l-4 ${
                      warning.severity === 'high' 
                        ? 'bg-red-50 border-red-400 text-red-800'
                        : warning.severity === 'medium'
                        ? 'bg-amber-50 border-amber-400 text-amber-800'
                        : 'bg-blue-50 border-blue-400 text-blue-800'
                    }`}
                  >
                    <div className="flex items-center mb-2">
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                      </svg>
                      <span className="font-semibold text-sm uppercase tracking-wide">{warning.type} - {warning.severity}</span>
                    </div>
                    <p className="text-sm leading-relaxed">{warning.message}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Main Content */}
            <div className="prose prose-gray max-w-none">
              <div className={`leading-relaxed whitespace-pre-wrap text-base ${
                isDark ? 'text-gray-200' : 'text-gray-800'
              }`}>
                {message.content}
              </div>
            </div>

            {/* Metadata Section */}
            {message.metadata && (
              <div className="mt-6 pt-6 border-t border-gray-100 space-y-6">
                
                {/* Sources */}
                {message.metadata.sources && message.metadata.sources.length > 0 && (
                  <div>
                    <div className="flex items-center mb-4">
                      <svg className="w-5 h-5 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                      <h4 className="font-semibold text-gray-800">Medical Sources</h4>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {message.metadata.sources.map((source: string, idx: number) => (
                        <div key={idx} className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-2 text-sm">
                          <span className="font-medium text-blue-800">
                            {source.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-3 italic">
                      Information verified from {message.metadata.sources.length} medical database{message.metadata.sources.length > 1 ? 's' : ''}
                    </p>
                  </div>
                )}

                {/* Confidence */}
                {message.metadata.confidence && (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Confidence Level</span>
                      <span className="text-sm font-bold text-emerald-600">{(message.metadata.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-emerald-500 to-green-500 h-2 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${message.metadata.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {/* Follow-up Questions */}
                {message.metadata.followUpQuestions && message.metadata.followUpQuestions.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                      <svg className="w-5 h-5 text-purple-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Related Questions
                    </h4>
                    <div className="space-y-2">
                      {message.metadata.followUpQuestions.map((question: string, idx: number) => (
                        <button
                          key={idx}
                          onClick={() => onFollowUpClick(question)}
                          className="w-full text-left bg-purple-50 hover:bg-purple-100 border border-purple-200 rounded-lg px-4 py-3 text-sm text-purple-800 transition-all duration-200 hover:shadow-md transform hover:-translate-y-0.5"
                        >
                          <div className="flex items-center justify-between">
                            <span>{question}</span>
                            <svg className="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default function ChatInterface({ sessionId: initialSessionId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(initialSessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    if (sessionId) {
      loadChatHistory();
    }
  }, [sessionId]);

  // Auto-resize effect for when input value changes programmatically
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      const scrollHeight = inputRef.current.scrollHeight;
      const maxHeight = 120;
      const minHeight = 56;
      
      if (scrollHeight > maxHeight) {
        inputRef.current.style.height = `${maxHeight}px`;
        inputRef.current.style.overflowY = 'auto';
      } else {
        inputRef.current.style.height = `${Math.max(scrollHeight, minHeight)}px`;
        inputRef.current.style.overflowY = 'hidden';
      }
    }
  }, [input]);

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`${getApiUrl(API_ENDPOINTS.chat)}?sessionId=${sessionId}`);
      const result = await response.json();
      
      if (result.success) {
        setMessages(result.data.history);
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const handleExampleClick = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  const handleFollowUpClick = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      createdAt: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setInput('');

    try {
      const response = await fetch(getApiUrl(API_ENDPOINTS.chat), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          sessionId,
        }),
      });

      const result = await response.json();

      if (result.success) {
        const agentResponse: AgentResponse = result.data.response;
        
        const assistantMessage: ChatMessage = {
          id: Date.now().toString() + '_assistant',
          role: 'assistant',
          content: agentResponse.content,
          metadata: {
            reasoning: agentResponse.reasoning,
            warnings: agentResponse.warnings,
            sources: agentResponse.sources,
            confidence: agentResponse.confidence,
            followUpQuestions: agentResponse.followUpQuestions,
          },
          createdAt: new Date(),
        };

        setMessages(prev => [...prev, assistantMessage]);
        setSessionId(result.data.sessionId);
      } else {
        throw new Error(result.error || 'Failed to send message');
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: ChatMessage = {
        id: Date.now().toString() + '_error',
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        createdAt: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(e);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    
    // Auto-resize textarea
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      const scrollHeight = inputRef.current.scrollHeight;
      const maxHeight = 120; // Max 5 lines approximately
      const minHeight = 56; // Min height to match design
      
      if (scrollHeight > maxHeight) {
        inputRef.current.style.height = `${maxHeight}px`;
        inputRef.current.style.overflowY = 'auto';
      } else {
        inputRef.current.style.height = `${Math.max(scrollHeight, minHeight)}px`;
        inputRef.current.style.overflowY = 'hidden';
      }
    }
  };

  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const hasConversation = messages.length > 0 || isLoading;

  return (
    <div className={`flex flex-col h-screen transition-colors duration-300 ${
      isDark ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      {/* Dynamic Header - At top when conversation starts */}
      {hasConversation && (
        <div className={`flex-shrink-0 max-w-5xl mx-auto w-full px-4 lg:px-6 ${
          isDark ? 'bg-gray-900/80' : 'bg-white/80'
        } backdrop-blur-lg`}>
          <WelcomeHeader isCompact={true} />
        </div>
      )}

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 lg:px-6 max-w-5xl mx-auto w-full">
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] px-6">
            <div className="mb-12">
              <WelcomeHeader isCompact={false} />
            </div>
            <ExampleCards onExampleClick={handleExampleClick} />
          </div>
        ) : (
          <div className="space-y-6 pt-6">
            {messages.map((message) => (
              <MessageBubble 
                key={message.id} 
                message={message} 
                onFollowUpClick={handleFollowUpClick}
              />
            ))}
          </div>
        )}

        {isLoading && (
          <div className="flex justify-start mb-8 animate-slide-in-left pt-6">
            <div className="max-w-4xl w-full">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-md">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div className={`rounded-2xl rounded-tl-md px-6 py-5 shadow-sm ${
                  isDark 
                    ? 'bg-gray-800 border border-gray-700' 
                    : 'bg-white border border-gray-200'
                }`}>
                  <ThinkingAnimation />
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className={`border-t backdrop-blur-lg px-4 lg:px-6 py-4 ${
        isDark 
          ? 'border-gray-700 bg-gray-900/80' 
          : 'border-gray-200 bg-white/80'
      }`}>
        <div className="max-w-5xl mx-auto">
          <form onSubmit={sendMessage} className="relative">
            <div className="flex items-end space-x-3">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about drug interactions, medical safety, or contraindications..."
                  disabled={isLoading}
                  rows={1}
                  className={`w-full resize-none rounded-2xl border px-6 py-4 focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:opacity-50 disabled:cursor-not-allowed text-base leading-6 transition-all duration-200 ${
                    isDark
                      ? 'border-gray-600 bg-gray-800 text-gray-100 placeholder-gray-400 focus:border-blue-400'
                      : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:border-blue-500'
                  }`}
                  style={{ 
                    resize: 'none',
                    scrollbarWidth: 'thin',
                    scrollbarColor: isDark ? '#4B5563 transparent' : '#CBD5E1 transparent',
                    minHeight: '56px',
                    maxHeight: '120px',
                    overflowY: 'hidden'
                  }}
                />
              </div>
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="flex-shrink-0 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-400 text-white rounded-2xl px-6 py-4 font-medium transition-all duration-200 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 disabled:transform-none disabled:shadow-none"
              >
                {isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                    <span>Sending</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                    <span>Send</span>
                  </div>
                )}
              </button>
            </div>
          </form>
          
          <div className="mt-4 text-center">
            <p className={`text-xs flex items-center justify-center ${
              isDark ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <svg className="w-4 h-4 mr-1 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              For informational purposes only. Always consult healthcare providers for medical advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}