import React from "react";
import { Grid, Paper } from "@mui/material";

const ImageGrid = ({ imageUrls }) => {
  return (
    <Grid container spacing={2}>
      {imageUrls.map((imageUrl, index) => (
        <Grid item key={index} xs={6} sm={4} md={3} lg={2}>
          <Paper elevation={3} style={{ padding: "10px", textAlign: "center" }}>
            <img
              src={imageUrl}
              alt={`Image ${index}`}
              style={{ maxWidth: "100%" }}
            />
          </Paper>
        </Grid>
      ))}
    </Grid>
  );
};

export default ImageGrid;
