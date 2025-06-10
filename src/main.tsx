import "./index.css"
import App from "./App.tsx"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { AntdConfigProvider } from "./config/antd.tsx"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AntdConfigProvider>
      <App />
    </AntdConfigProvider>
  </StrictMode>
)
