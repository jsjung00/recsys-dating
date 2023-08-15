import React from "react";
import { Card, CardMedia, CircularProgress } from "@mui/material";

const ImageCard = ({ imageUrl, sx, className, isLoading, childComp }) => {
  return (
    <Card raised className={className} sx={{ ...sx }}>
      {isLoading && <CircularProgress style={{ position: "absolute" }} />}
      <CardMedia
        component={"img"}
        image={imageUrl}
        sx={{
          objectFit: "contain",
          height: "100%",
          opacity: isLoading && "0.3",
          filter: isLoading && "blur(1.3rem)",
          loading: "lazy",
        }}
      ></CardMedia>
    </Card>
  );
};

export default ImageCard;
