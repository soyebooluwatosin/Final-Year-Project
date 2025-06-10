import { createBrowserRouter } from "react-router-dom"
import ErrorPage from "../pages/ErrorPage"
import Analysis from "../pages/Analysis"
import Landing from "../pages/Landing"
import Login from "../pages/Login"
import Signup from "../pages/Signup"

// Define all application routes
export const router = createBrowserRouter([
  {
    path: "/",
    element: <Landing />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/login",
    element: <Login />,
  },
  {
    path: "/signup",
    element: <Signup />,
  },
  {
    path: "/analysis",
    element: <Analysis />,
  },
])
