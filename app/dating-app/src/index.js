import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Root from "./routes/root";
import Results from "./routes/results";
import { ObjectProvider } from "./context";
import "./index.css";
import reportWebVitals from "./reportWebVitals";
import Rating from "./routes/rating";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
  },
  {
    path: "/rating",
    element: <Rating />,
  },
  {
    path: "/results",
    element: <Results />,
  },
]);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <ObjectProvider>
    <RouterProvider router={router} />
  </ObjectProvider>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
