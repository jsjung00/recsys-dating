import React, { useEffect, useState, useRef, useMemo } from "react";
import TinderCard from "react-tinder-card";
import "./TinderCards.css";
import database from "./firebase";
import { collection, getDocs } from "firebase/firestore";
import ReplayIcon from "@mui/icons-material/Replay";
import CloseIcon from "@mui/icons-material/Close";
import FavoriteIcon from "@mui/icons-material/Favorite";
import IconButton from "@mui/material/IconButton";
//TODO: add loading behind all the cards

export default function TinderCards(props) {
  const { people, setPeople, afterFeedback } = props;
  console.log("people", people);
  const canSwipe = people.length > 0;

  const swiped = (direction) => {
    //card to be deleted is the last element in people array
    const last_person = people[people.length - 1];
    console.log("last person", last_person);
    afterFeedback(
      direction === "right",
      last_person.id,
      last_person.cluster_idx
    );
    const new_people = people.slice(0, -1);
    setPeople(new_people);
  };

  return (
    <>
      <div className="tinderCards__cardContainer">
        {people.map((person) => (
          <TinderCard
            className="swipe"
            key={person.id}
            preventSwipe={["up", "down"]}
            onSwipe={(dir) => swiped(dir)}
          >
            <div
              style={{ backgroundImage: `url(${person.url})` }}
              className="card"
            >
              <h1 style={{ color: "white" }}>{person.id}</h1>
            </div>
          </TinderCard>
        ))}
      </div>
      <div className="swipeButtons">
        <IconButton
          className="swipeButtons__close"
          disabled={!canSwipe}
          onClick={() => swiped("left")}
        >
          <CloseIcon fontsize="large" />
        </IconButton>
        <IconButton
          className="swipeButtons__favorite"
          disabled={!canSwipe}
          onClick={() => swiped("right")}
        >
          <FavoriteIcon fontsize="large" />
        </IconButton>
      </div>
    </>
  );
}
