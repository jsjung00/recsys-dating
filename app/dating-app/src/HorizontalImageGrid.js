import React from "react";
import {
  Container,
  ImageList,
  ImageListItem,
  Card,
  CardMedia,
  Grid,
} from "@mui/material";

const HorizontalImageGrid = ({ images, imageHeight }) => {
  return (
    <div style={{ display: "flex", overflowX: "auto" }}>
      {images.map((image, index) => (
        <div key={index} style={{ flex: "0 0 auto", padding: "0 8px" }}>
          <Card raised sx={{ height: imageHeight }}>
            <CardMedia
              component={"img"}
              image={image.url}
              sx={{
                objectFit: "contain",
                height: "100%",
              }}
            ></CardMedia>
          </Card>
        </div>
      ))}
    </div>
  );
};

export default HorizontalImageGrid;
