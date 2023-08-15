import * as React from "react";
import ManIcon from "@mui/icons-material/Man";
import WomanIcon from "@mui/icons-material/Woman";
import WcIcon from "@mui/icons-material/Wc";
import {
  Grid,
  Paper,
  Card,
  CardContent,
  IconButton,
  CardMedia,
  Typography,
  Button,
  Container,
  CardActionArea,
} from "@mui/material";
import { redirect, useNavigate } from "react-router-dom";

export default function Root() {
  const navigate = useNavigate();
  const handleGenderChange = (newGender) => {
    //move to rating page
    navigate("/rating", {
      state: { gender: newGender },
    });
  };

  return (
    <Container
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100%", // Take up full viewport height
        padding: "5%", // Add padding
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          textAlign: "center",
          marginBottom: "10%",
        }}
      >
        <h1>What's Your Type?</h1>a content-based recommender system for dating
      </div>

      <h2 style={{ marginBottom: "20px" }}>Choose your interest</h2>
      <Grid
        container
        spacing={2}
        justifyContent={"center"}
        alignItems={"center"}
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        {[
          // Map over an array to generate the cards
          {
            gender: "man",
            label: "Male",
            icon: <ManIcon sx={{ fontSize: "15vh" }} />,
          },
          {
            gender: "woman",
            label: "Female",
            icon: <WomanIcon sx={{ fontSize: "15vh" }} />,
          },
          {
            gender: "both",
            label: "Both",
            icon: <WcIcon sx={{ fontSize: "15vh" }} />,
          },
        ].map((item) => (
          <Grid key={item.gender} item xs={3} sm={3} md={3} lg={3}>
            <Card>
              <CardActionArea
                onClick={() => handleGenderChange(item.gender)}
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  height: "100%", // Occupy the entire height of the grid cell
                }}
              >
                {item.icon}
                <CardContent style={{ textAlign: "center" }}>
                  <Typography gutterBottom variant="h5" component="div">
                    {item.label}
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}
