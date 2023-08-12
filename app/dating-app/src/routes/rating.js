import React, { useEffect, useState, useRef } from "react";
import TinderCards from "../TinderCards";
import database from "../firebase";
import { doc, getDoc } from "firebase/firestore";
import GENDER from "../variables";
//TODO (JJ): get gender from the user selection
//TODO: create image ID and URL objects

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
    let ret_image = null;
    while (ret_image === null) {
      const random_id = Math.floor(Math.random() * this.image_ids.length);
      if (seen_images.has(random_id)) {
        continue;
      } else {
        ret_image = this.image_ids[random_id];
      }
    }
    return ret_image;
  }

  update_values(rating, image_id) {
    if (rating === 1) {
      this.numLikes += 1;
      this.likedIds.push(image_id);
    } else {
      this.numDislikes += 1;
      this.dislikedIds.push(image_id);
    }
  }

  print_stats() {
    console.log(`Number of likes ${this.numLikes}`);
    console.log(`Number of dislikes ${this.numDislikes}`);
  }
}

export default function Rating() {
  const [IDToURLMap, setIDToURLMap] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [imageIDURL, setImageIDURL] = useState([]); //contains objects {id: , url:, cluster_idx:}
  const [totalLikes, setTotalLikes] = useState(0);
  const [roundNumber, setRoundNumber] = useState(0);

  async function initClusters() {
    console.log("called initClusters");
    let doc_name;
    if (GENDER === "FEMALE") {
      doc_name = "TMDBID_KMEANS_k=10_female_5000_5-5_nonan";
    } else if (GENDER === "MALE") {
      doc_name = "TMDBID_KMEANS_k=10_male_5000_5-5_nonan";
    } else {
      //pass
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

  async function initIDToURLMap() {
    console.log("Called initIDToURLMap");
    const map_doc = doc(database, "misc", "tmdbIDToURL");
    const map_snap = await getDoc(map_doc);
    if (map_snap.exists()) {
      const IDToURLMap = map_snap.data();
      setIDToURLMap(IDToURLMap);
      return IDToURLMap;
    } else {
      //failed to get document
      console.log("handle error to get clusters");
    }
  }

  function generateInitialStack(cluster_arr, url_map) {
    //samples one from each cluster to create initial stack
    let initStack = [];
    console.log("clusters in generateInitStack", cluster_arr);
    for (let i = 0; i < cluster_arr.length; i++) {
      const imageID = cluster_arr[i].getRandomImageId();
      initStack.push({ id: imageID, url: url_map[imageID], cluster_idx: i });
    }
    console.log("generate init stack", initStack);
    return initStack;
  }

  function afterFeedback(rating, imageID, clusterIdx) {
    //rating (int) 1 or 0
    clusters[clusterIdx].update_values(rating, imageID);
    setRoundNumber((roundNumber) => roundNumber + 1);
  }

  useEffect(() => {
    if (roundNumber > 0) {
      executeRound();
    }
  }, [roundNumber]); //called after roundNumber updates

  function executeRound() {
    console.log("Finished round", roundNumber);

    if (roundNumber < clusters.length) {
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
        { id: imageID, url: IDToURLMap[imageID], cluster_idx: cluster_idx },
      ]);
      console.log("Added new object");
    }
  }

  async function bootup() {
    const urlMap = await initIDToURLMap();
    const cluster_arr = await initClusters();
    console.log("cluster_arr para", cluster_arr);
    const init_stack = generateInitialStack(cluster_arr, urlMap);
    setImageIDURL(init_stack);
  }
  useEffect(() => {
    bootup();
  }, []);

  return (
    <>
      <TinderCards
        people={imageIDURL}
        setPeople={setImageIDURL}
        afterFeedback={afterFeedback}
      />
    </>
  );
}
