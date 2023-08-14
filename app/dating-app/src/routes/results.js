import React, { useContext, useEffect, useState, useRef } from "react";
import { ObjectContext } from "../context";
import { useLocation } from "react-router-dom";
import database from "../firebase";
import { doc, setDoc, onSnapshot } from "firebase/firestore";
import { Container } from "@mui/material";
import ImageGrid from "../ImageGrid";

export default function Results() {
  const { urlMap } = useContext(ObjectContext);
  const location = useLocation();
  const { likedIds, dislikedIds } = location.state;
  const [recommendedPeople, setRecommendedPeople] = useState([]);

  async function upload_rating_data() {
    const data = {
      bucket: "similarity-matix",
      sim_file: "simMatrix_both_5000_nonan.npy",
      sim_id_file: "tmdb_ids_both_5000_nonan.npy",
      liked_ids: likedIds,
      disliked_ids: dislikedIds,
      cluster_size: 6,
      sim_threshold: 0.75,
    };

    //const document_name = (Math.random() + 1).toString(36).substring(2);
    //await setDoc(doc(database, "users-ratings", document_name), data);
    const get_rec_url =
      "https://us-central1-dating-recsys.cloudfunctions.net/return_rec_ids";
    fetch(get_rec_url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.text();
      })
      .then((data) => {
        const recIDs = data.split(",");
        const recPeople = recIDs.map((ID) => {
          const retObject = {
            id: ID,
            url: urlMap[ID],
          };
          return retObject;
        });
        setRecommendedPeople(recPeople);
        console.log("Successfully received recommended IDs", data); // Success message from the Cloud Function
      })
      .catch((error) => {
        console.error("There was a problem with the fetch operation:", error);
      });
  }

  useEffect(() => {
    if (recommendedPeople.length > 0) {
      //refresh to display recommended images
    }
  }, [recommendedPeople]);

  useEffect(() => {
    upload_rating_data();
    //get_recommended_ids();
  }, []);

  return (
    <>
      {recommendedPeople.length > 0 ? (
        <Container>
          <h1>Recommended Images </h1>
          <ImageGrid imageUrls={recommendedPeople.map((obj) => obj.url)} />
        </Container>
      ) : (
        <p>Loading (TODO: Add loading component)</p>
      )}
    </>
  );
}
