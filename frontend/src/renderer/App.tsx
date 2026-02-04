import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import NoteClusters from './NoteClusters';
import './App.css';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<NoteClusters />} />
      </Routes>
    </Router>
  );
}
