import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Checkbox from "@mui/material/Checkbox";
import DoneIcon from "@mui/icons-material/Done";

const columns = [
  {
    id: "cnn1_annotations_exist",
    label: "Annotations",
    minWidth: 140,
    align: "center",
    format: (value) => value.toString(),
  },
  {
    id: "cnn1_predictions_exist",
    label: "Predictions",
    minWidth: 140,
    align: "center",
    format: (value) => value.toString(),
  },
  { id: "tileEdge", label: "Tile", minWidth: 80, align: "center" },
  {
    id: "timestamp",
    label: "Upload",
    minWidth: 170,
    align: "center",
    format: (value) => new Date(value).toLocaleString("en-GB"),
  },
  {
    id: "updatedAt",
    label: "Last Updated",
    minWidth: 170,
    align: "center",
    format: (value) => new Date(value).toLocaleString("en-GB"),
  },
  {
    id: "enabled",
    label: "Enabled",
    minWidth: 80,
    align: "center",
    format: (value) => value.toString(),
  },
];

const PredictionAndAnnotationTable = ({
  data,
  selectedKey,
  updateSelectedRow,
}) => {
  const handleClick = (key, tileEdge) => {
    if (key !== selectedKey) {
      updateSelectedRow(key, tileEdge);
    } else {
      updateSelectedRow(null, null);
    }
  };

  return (
    <Paper
      sx={{
        width: "100%",
        overflow: "hidden",
        borderStyle: "solid",
        borderWidth: "1px",
        borderColor: "lightGray",
      }}
    >
      <TableContainer sx={{ maxHeight: 400 }}>
        <Table stickyHeader aria-label="sticky table">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox"></TableCell>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(data).map(([key, row]) => {
              const isRowSelected = selectedKey === key;
              return (
                <TableRow
                  hover
                  key={row.timestamp}
                  selected={isRowSelected}
                  onClick={() => handleClick(key, row.tileEdge)}
                >
                  <TableCell padding="checkbox">
                    <Checkbox color="primary" checked={isRowSelected} />
                  </TableCell>
                  {columns.map((column) => {
                    const value = row[column.id];
                    return (
                      <TableCell key={column.id} align={column.align}>
                        {typeof value == "boolean" && value && (
                          <DoneIcon className="annotationAndPredictionTableTick" />
                        )}
                        {typeof value != "boolean" &&
                          (column.format ? column.format(value) : value)}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default PredictionAndAnnotationTable;
