import React, { useContext, useEffect, useState, useRef } from "react";
import { ObjectContext } from "../context";
import { useLocation } from "react-router-dom";
import { Container, Typography } from "@mui/material";
import ImageGrid from "../ImageGrid";
import HorizontalImageGrid from "../HorizontalImageGrid";
import LinearProgress from "@mui/material/LinearProgress";
import Box from "@mui/material/Box";

export default function Results() {
  const { urlMap } = useContext(ObjectContext);
  const location = useLocation();
  const { likedIds, dislikedIds, gender } = location.state;
  const [recommendedPeople, setRecommendedPeople] = useState([]);

  async function upload_rating_data() {
    let data = {
      bucket: "similarity-matix",
      liked_ids: likedIds,
      disliked_ids: dislikedIds,
      cluster_size: 6,
      sim_threshold: 0.75,
    };

    if (gender === "man") {
      data.sim_file = "simMatrix_male_5000_5-5_nonan.npy";
      data.sim_id_file = "tmdb_ids_male_5000_nonan.npy";
    } else if (gender === "woman") {
      data.sim_file = "simMatrix_female_5000_5-5_nonan.npy";
      data.sim_id_file = "tmdb_ids_female_5000_nonan.npy";
    } else if (gender === "both") {
      data.sim_file = "simMatrix_both_5000_nonan.npy";
      data.sim_id_file = "tmdb_ids_both_5000_nonan.npy";
    } else {
      console.error("Received incorrect gender from /rating");
    }

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
      console.log(recommendedPeople);
    }
  }, [recommendedPeople]);

  useEffect(() => {
    upload_rating_data();
    //get_recommended_ids();
  }, []);

  return (
    <Container sx={{ padding: "2rem" }}>
      {recommendedPeople.length > 0 ? (
        <Container
          sx={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            textAlign: "center",
            padding: "5vh",
            marginTop: "25vh",
            marginBottom: "10vh",
          }}
        >
          <h1 style={{ margin: "2rem" }}>Recommended Category: </h1>
          <ImageGrid imageUrls={recommendedPeople.map((obj) => obj.url)} />
        </Container>
      ) : (
        <Container
          sx={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            padding: "5vh",
            marginTop: "25vh",
            marginBottom: "10vh",
            textAlign: "center",
          }}
        >
          <Typography sx={{ padding: "1rem" }}>
            {"Generating recommended type... (ETA: < 1 min)"}{" "}
          </Typography>
          <LinearProgress />
        </Container>
      )}
      <Container
        sx={{
          textAlign: "center",
        }}
      >
        <h2 style={{ margin: "1rem" }}>Liked Images </h2>
        <HorizontalImageGrid
          images={likedIds.map((id) => {
            const newObj = { id: id, url: urlMap[id] };
            return newObj;
          })}
          imageHeight={"20vh"}
        />
      </Container>
      <Container
        sx={{
          textAlign: "center",
        }}
      >
        <h2 style={{ margin: "1rem" }}>Disliked Images </h2>
        <HorizontalImageGrid
          images={dislikedIds.map((id) => {
            const newObj = { id: id, url: urlMap[id] };
            return newObj;
          })}
          imageHeight={"20vh"}
        />
      </Container>
    </Container>
  );
}
