import { createTheme, responsiveFontSizes } from "@mui/material/styles";

const nonResponsiveTheme = createTheme();
export const theme = responsiveFontSizes(nonResponsiveTheme);
export const GENDER = "FEMALE";
export const MIN_ROUNDS = 30;
export const MIN_LIKES = 5;
