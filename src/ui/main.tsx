import React, { Component, type ErrorInfo, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'
import './light.css'
import './expressive.css'
import './final-polish.css'

class RootBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Frontend render error:', error, info.componentStack) }
  render() {
    if (this.state.error) return <main className="runtime-error"><span className="eyebrow">INTERFACE ERROR</span><h1>This view could not render</h1><p>{this.state.error.message}</p><button className="primary-btn" onClick={() => location.reload()}>Reload terminal</button></main>
    return this.props.children
  }
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><RootBoundary><App /></RootBoundary></React.StrictMode>)
