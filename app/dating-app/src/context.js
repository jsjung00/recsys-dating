import { createContext, useEffect, useState } from "react";
import database from "./firebase";
import { doc, getDoc } from "firebase/firestore";
export const ObjectContext = createContext();

export const ObjectProvider = ({ children }) => {
  const [urlMap, setUrlMap] = useState({});
  useEffect(() => {
    initIDToURLMap();
  }, []);

  async function initIDToURLMap() {
    const map_doc = doc(database, "misc", "tmdbIDToURL");
    const map_snap = await getDoc(map_doc);
    if (map_snap.exists()) {
      const IDToURLMap = map_snap.data();
      setUrlMap(IDToURLMap);
    } else {
      //failed to get document
      console.log("handle error to get clusters");
    }
  }

  return (
    <ObjectContext.Provider value={{ urlMap, setUrlMap }}>
      {children}
    </ObjectContext.Provider>
  );
};
