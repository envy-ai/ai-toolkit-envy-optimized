// ErrorBoundary.tsx
'use client';

import React, { ReactNode, ErrorInfo, Component } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
  componentStack: string;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null, componentStack: '' };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
    this.setState({ componentStack: errorInfo.componentStack || '' });
  }

  render(): ReactNode {
    const { error, componentStack } = this.state;
    if (error) {
      return (
        <div role="alert" className="rounded border border-red-700 bg-red-950/40 p-4 text-red-200">
          {this.props.fallback || <p>Unable to display this section.</p>}
          <p className="mt-2 font-semibold whitespace-pre-wrap break-words">{error.message}</p>
          <details className="mt-2">
            <summary className="cursor-pointer">Error details and stack trace</summary>
            <pre className="mt-2 overflow-auto whitespace-pre-wrap break-words text-xs">
              {error.stack || `${error.name}: ${error.message}`}
              {componentStack && `\n\nComponent stack:${componentStack}`}
            </pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
