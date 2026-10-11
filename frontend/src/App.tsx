import { BrowserRouter, Routes, Route } from "react-router-dom";
import Workspace from "./pages/Workspace";
import Results from "./pages/Results";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Workspace />} />
        <Route path="/results/:id" element={<Results />} />
      </Routes>
    </BrowserRouter>
  );
}
