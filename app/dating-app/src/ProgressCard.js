import React from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import LinearProgress from "@mui/material/LinearProgress";
import Typography from "@mui/material/Typography";
import { theme } from "./variables";
import { ThemeProvider } from "@mui/material";

const ProgressCard = ({ roundNumber, likes, style }) => {
  return (
    <Card sx={{ ...style }}>
      <CardContent>
        <ThemeProvider theme={theme}>
          <Typography
            variant="caption"
            display="block"
            align="right"
            gutterBottom
          >
            Round:{" "}
            <Typography variant="h5" display="inline">
              {roundNumber}/30
            </Typography>
          </Typography>
          <Typography
            variant="caption"
            display="block"
            align="right"
            gutterBottom
          >
            Likes:{" "}
            <Typography variant="h5" display="inline">
              {likes}/6
            </Typography>
          </Typography>
        </ThemeProvider>
      </CardContent>
    </Card>
  );
};

export default ProgressCard;
