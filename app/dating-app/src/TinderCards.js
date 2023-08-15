import React, { useEffect, useState, useRef, useMemo } from "react";
import TinderCard from "react-tinder-card";
import "./TinderCards.css";
import ImageCard from "./ImageCard";
import CloseIcon from "@mui/icons-material/Close";
import FavoriteIcon from "@mui/icons-material/Favorite";
import IconButton from "@mui/material/IconButton";
import { Card, CircularProgress, Container } from "@mui/material";
//TODO: add loading behind all the cards

export default function TinderCards(props) {
  const { people, setPeople, afterFeedback } = props;
  const canSwipe = people.length > 0;
  const maxCardWidth = Math.min(0.5 * window.innerWidth, 300);

  const swiped = (direction) => {
    //card to be deleted is the first element in people array
    const first_person = people[0];
    afterFeedback(
      direction === "right",
      first_person.id,
      first_person.cluster_idx
    );

    const new_people = people.slice(1);
    console.log(new_people);
    setPeople(new_people);
  };

  return (
    <div
      id="parent-container"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        height: "100%",
        width: "100%",
      }}
    >
      <div
        id="card-container"
        style={{
          height: "60%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          position: "relative",
          marginTop: "5%",
        }}
      >
        {people.length > 0 ? (
          people.map((person, index) => (
            <ImageCard
              className={`${index === 0 ? "topCard" : ""}`}
              sx={{
                position: "absolute",
                left: "50%",
                transform: "translateX(-50%)",
                maxWidth: "70vw",
                height: "100%",
                opacity: index === 0 ? 1 : 0, // Adjust opacity to create the stacking effect
                transition: "opacity 0.3s ease-in-out", // Smooth opacity transition
              }}
              key={person.id}
              imageUrl={person.url}
            ></ImageCard>
          ))
        ) : (
          <ImageCard
            sx={{
              position: "absolute",
              left: "50%",
              transform: "translateX(-50%)",
              maxWidth: "70vw",
              height: "100%",
              filter: "blur(0rem)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
            isLoading={true}
            key={"loading-card"}
            imageUrl={
              "https://www.themoviedb.org/t/p/original/xxYawgFO1woBRveH7WL9D1BxB4W.jpg"
            }
          ></ImageCard>
        )}
      </div>
      <div style={{ flexGrow: 1 }}></div>
      <Container
        sx={{
          width: "50%",
          display: "flex",
          justifyContent: "space-evenly",
          marginBottom: "3rem",
        }}
        className="swipeButtons"
      >
        <IconButton
          className="swipeButtons__close"
          disabled={!canSwipe}
          onClick={() => swiped("left")}
          sx={{ width: "10vh", height: "10vh", fontSize: "6vh" }}
        >
          <CloseIcon fontSize="inherit" />
        </IconButton>
        <IconButton
          className="swipeButtons__favorite"
          disabled={!canSwipe}
          onClick={() => swiped("right")}
          sx={{ width: "10vh", height: "10vh", fontSize: "6vh" }}
        >
          <FavoriteIcon sx={{ fontSize: "inherit" }} />
        </IconButton>
      </Container>
    </div>
  );
}
