import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyChZ-FUh5rLAbI-BAQ20cqbUBrRmsFb8Ls",
  authDomain: "dating-recsys.firebaseapp.com",
  projectId: "dating-recsys",
  storageBucket: "dating-recsys.appspot.com",
  messagingSenderId: "891585968862",
  appId: "1:891585968862:web:d375fab10a30e04546604f",
  measurementId: "G-0BNY1MXFYC",
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const database = getFirestore(firebaseApp);

export default database;
