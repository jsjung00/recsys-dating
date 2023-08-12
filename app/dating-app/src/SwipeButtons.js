import React from "react";
import ReplayIcon from "@mui/icons-material/Replay";
import CloseIcon from "@mui/icons-material/Close";
import FavoriteIcon from "@mui/icons-material/Favorite";
import IconButton from "@mui/material/IconButton";

import "./SwipeButtons.css";

const SwipeButtons = () => {
  return (
    <div className="swipeButtons">
      <IconButton className="swipeButtons__close">
        <CloseIcon fontsize="large" />
      </IconButton>
      <IconButton className="swipeButtons__repeat">
        <ReplayIcon fontsize="large" />
      </IconButton>
      <IconButton className="swipeButtons__favorite">
        <FavoriteIcon fontsize="large" />
      </IconButton>
    </div>
  );
};

export default SwipeButtons;
