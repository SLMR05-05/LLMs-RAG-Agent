import { Navigate, Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import NotebookPage from './pages/NotebookPage'
import './index.css'

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/notebook/:id" element={<NotebookPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
