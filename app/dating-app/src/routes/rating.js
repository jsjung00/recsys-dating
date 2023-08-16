import React, { useContext, useEffect, useState, useRef } from "react";
import { ObjectContext } from "../context";
import TinderCards from "../TinderCards";
import ProgressCard from "../ProgressCard";

import database from "../firebase";
import { doc, getDoc } from "firebase/firestore";
import { MIN_ROUNDS, MIN_LIKES } from "../variables";
import { redirect, useNavigate } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { Container } from "@mui/material";
//TODO: change tinder card to regular MUI card

class Cluster {
  constructor(image_ids) {
    this.image_ids = image_ids;
    this.numLikes = 0;
    this.numDislikes = 0;
    this.likedIds = [];
    this.dislikedIds = [];
  }

  getUCBScore(numRounds) {
    const avgRating = this.numLikes / (this.numDislikes + this.numLikes);
    const numVisits = this.numLikes + this.numDislikes;
    return avgRating + Math.sqrt(Math.log(numRounds) / numVisits);
  }

  getRandomImageId() {
    const seen_images = new Set([...this.likedIds, ...this.dislikedIds]);
    let ret_id = null;
    while (ret_id === null) {
      const random_idx = Math.floor(Math.random() * this.image_ids.length);
      const random_id = this.image_ids[random_idx];
      if (seen_images.has(random_id)) {
        continue;
      } else {
        ret_id = random_id;
      }
    }
    return ret_id;
  }

  update_values(rating, image_id) {
    if (rating === true) {
      this.numLikes += 1;
      this.likedIds.push(image_id);
    } else {
      this.numDislikes += 1;
      this.dislikedIds.push(image_id);
    }
  }

  print_stats() {
    console.log("Liked Ids", this.likedIds);
    console.log("Disliked Ids", this.dislikedIds);
  }
}

export default function Rating() {
  const { urlMap } = useContext(ObjectContext);
  const [clusters, setClusters] = useState([]);
  const [imageIDURL, setImageIDURL] = useState([]); //contains objects {id: , url:, cluster_idx:}
  const [totalLikes, setTotalLikes] = useState(0);
  const [likedIDs, setLikedIDs] = useState([]);
  const [dislikedIDs, setDislikedIDs] = useState([]);
  const [roundNumber, setRoundNumber] = useState(1);
  const navigate = useNavigate();
  const location = useLocation();
  const { gender } = location.state;

  async function initClusters() {
    let doc_name;
    if (gender === "woman") {
      doc_name = "TMDBID_KMEANS_k=10_female_5000_5-5_nonan";
    } else if (gender === "man") {
      doc_name = "TMDBID_KMEANS_k=10_male_5000_5-5_nonan";
    } else if (gender == "both") {
      doc_name = "TMDBID_KMEANS_k=20_both_5000_5-5_nonan";
    } else {
      console.error("Received incorrect gender option");
    }
    const cluster_doc = doc(database, "clusters", doc_name);
    const cluster_snap = await getDoc(cluster_doc);
    if (cluster_snap.exists()) {
      let cluster_arr = [];
      let cluster_data = cluster_snap.data();
      for (let key in cluster_data) {
        if (cluster_data.hasOwnProperty(key)) {
          const ids = cluster_data[key];
          const cluster = new Cluster(ids);
          cluster_arr.push(cluster);
        }
      }
      setClusters(cluster_arr);
      return cluster_arr;
    } else {
      //failed to get document
      console.log("handle error to get clusters");
    }
  }

  function generateInitialStack(cluster_arr, url_map) {
    //samples one from each cluster to create initial stack
    let initStack = [];
    for (let i = 0; i < cluster_arr.length; i++) {
      const imageID = cluster_arr[i].getRandomImageId();
      initStack.push({ id: imageID, url: url_map[imageID], cluster_idx: i });
    }
    return initStack;
  }

  async function afterFeedback(rating, imageID, clusterIdx) {
    //rating (bool)
    clusters[clusterIdx].update_values(rating, imageID);
    if (rating === true) {
      setTotalLikes((prevNumLikes) => prevNumLikes + 1);
      setLikedIDs([...likedIDs, imageID]);
    } else {
      setDislikedIDs([...dislikedIDs, imageID]);
    }

    console.log(`Updated cluster ${clusterIdx}`);
    for (let i = 0; i < clusters.length; i++) {
      console.log(`Cluster ${i + 1}`);
      clusters[i].print_stats();
    }
    setRoundNumber((roundNumber) => roundNumber + 1);
  }

  useEffect(() => {
    if (roundNumber > 1) {
      ExecuteRound();
    }
  }, [roundNumber, totalLikes, likedIDs, dislikedIDs]); //called after roundNumber updates

  function ExecuteRound() {
    console.log("Finished round", roundNumber);
    if (roundNumber > MIN_ROUNDS && totalLikes > MIN_LIKES) {
      //finish rating phase, move to results display
      navigate("/results", {
        state: { likedIds: likedIDs, dislikedIds: dislikedIDs, gender: gender },
      });
    }
    if (roundNumber <= clusters.length) {
      //pass since initial stack contains one from each cluster
    } else {
      //draw from UCB max cluster
      let imageID;
      let cluster_idx;
      for (let i = 0; i < clusters.length; i++) {
        const cluster = clusters[i];
        cluster.print_stats();
      }
      const cluster_ucb_values = clusters.map((cluster) =>
        cluster.getUCBScore(roundNumber)
      );
      const max_idx = cluster_ucb_values.indexOf(
        Math.max(...cluster_ucb_values)
      );
      imageID = clusters[max_idx].getRandomImageId();
      cluster_idx = max_idx;
      setImageIDURL([
        ...imageIDURL,
        { id: imageID, url: urlMap[imageID], cluster_idx: cluster_idx },
      ]);
      console.log("Added new object with ID", imageID);
    }
  }

  async function bootup() {
    const cluster_arr = await initClusters();
    const init_stack = generateInitialStack(cluster_arr, urlMap);
    console.log("init_stack", init_stack);
    setImageIDURL(init_stack);
  }
  useEffect(() => {
    if (Object.keys(urlMap).length > 0) {
      bootup();
    }
  }, [urlMap]);

  return (
    <div style={{ height: "100vh", width: "100vw", position: "relative" }}>
      <TinderCards
        people={imageIDURL}
        setPeople={setImageIDURL}
        afterFeedback={afterFeedback}
      />
      <div
        style={{
          position: "absolute",
          top: "4vh",
          right: "2vw",
          display: "inline-block",
        }}
      >
        <ProgressCard
          style={{ height: "auto", width: "auto" }}
          roundNumber={roundNumber}
          likes={totalLikes}
        />
      </div>
    </div>
  );
}
