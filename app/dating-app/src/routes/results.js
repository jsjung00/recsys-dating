import React, { useContext, useEffect, useState, useRef } from "react";
import { ObjectContext } from "../context";
import { useLocation } from "react-router-dom";
import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Paper,
  LinearProgress,
} from "@mui/material";
import ImageGrid from "../ImageGrid";
import HorizontalImageGrid from "../HorizontalImageGrid";

export default function Results() {
  const { urlMap } = useContext(ObjectContext);
  const location = useLocation();
  const { likedIds, dislikedIds, gender } = location.state;
  const [recommendedPeople, setRecommendedPeople] = useState([]);
  const [vizImageUrl, setVizImageUrl] = useState(null);

  async function upload_rating_data() {
    let data = {
      bucket: "similarity-matix",
      liked_ids: likedIds,
      disliked_ids: dislikedIds,
      cluster_size: 6,
      sim_threshold: 0.75,
    };
    console.log("LikedIDS", likedIds);
    console.log("Disliked Ids", dislikedIds);

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

  async function get_image_viz() {
    let data = {
      bucket: "embedding-files",
      liked_ids: likedIds,
      disliked_ids: dislikedIds,
    };

    if (gender === "man") {
      data.id_idx_map_file = "tmdb_ids_male_5000_nonan.npy";
      data.tsne_embed_file = "tsneperp=5_emb_matrix_male_5000_5-5_nonan.npy";
    } else if (gender === "woman") {
      data.id_idx_map_file = "tmdb_ids_female_5000_nonan.npy";
      data.tsne_embed_file = "tsneperp=5_emb_matrix_female_5000_5-5_nonan.npy";
    } else if (gender === "both") {
      data.id_idx_map_file = "tmdb_ids_both_5000_nonan.npy";
      data.tsne_embed_file = "tsneperp=5_emb_matrix_both_5000_5-5_nonan.npy";
    } else {
      console.error("Received incorrect gender from /rating");
    }

    const get_rec_url =
      "https://us-central1-dating-recsys.cloudfunctions.net/send_image_viz";
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
        const received_obj = JSON.parse(data);
        const encodedImg = received_obj.image;
        const imageUrl = `data:image/png;base64,${encodedImg}`;
        setVizImageUrl(imageUrl);
        console.log("Successfully set visualize URL", imageUrl); // Success message from the Cloud Function
      })
      .catch((error) => {
        console.error("There was a problem with the fetch operation:", error);
      });
  }

  useEffect(() => {
    //refresh to display recommended images or visualized image
    if (recommendedPeople.length > 0) {
      console.log(recommendedPeople);
    }
  }, [recommendedPeople, vizImageUrl]);

  useEffect(() => {
    upload_rating_data();
    get_image_viz();
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
      <Container sx={{ textAlign: "center" }}>
        <h2 style={{ margin: "1rem" }}>Ratings Visualized </h2>
        {vizImageUrl != null ? (
          <Paper elevation={3} style={{ padding: 0, display: "inline-block" }}>
            <img
              src={vizImageUrl}
              alt="Base64 Image"
              style={{ height: "45vh", width: "auto", display: "block" }}
            />
          </Paper>
        ) : (
          <Paper elevation={3} style={{ padding: 20 }}>
            <CircularProgress />
          </Paper>
        )}
      </Container>
    </Container>
  );
}
