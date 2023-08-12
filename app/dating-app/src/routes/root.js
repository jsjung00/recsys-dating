import * as React from "react";
import ManIcon from "@mui/icons-material/Man";
import WomanIcon from "@mui/icons-material/Woman";
import WcIcon from "@mui/icons-material/Wc";
import Stack from "@mui/material/Stack";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";

export default function Root() {
  const [gender, setGender] = React.useState(() => ["male"]);
  const handleGender = (event, newGender) => {
    if (newGender.length) {
      setGender(newGender);
    }
  };

  return (
    <>
      <div id="selection-container">
        <Stack direction="row" spacing={4}>
          <ToggleButtonGroup
            exclusive
            value={gender}
            onChange={handleGender}
            aria-label="gender"
          >
            <ToggleButton value="male" aria-label="male">
              <ManIcon />
            </ToggleButton>
            <ToggleButton value="female" aria-label="female">
              <WomanIcon />
            </ToggleButton>
            <ToggleButton value="both" aria-label="both">
              <WcIcon />
            </ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </div>
    </>
  );
}
