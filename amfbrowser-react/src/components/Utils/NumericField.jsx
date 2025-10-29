import { useId } from "react";
import { NumberField } from "@base-ui-components/react/number-field";
import "./styles/NumericFieldStyles.css";

export default function NumericField({
  label,
  value,
  onChange,
  min = 0,
  max = undefined,
  step = 1,
}) {
  const id = useId();

  return (
    <NumberField.Root
      id={id}
      value={value}
      defaultValue={0}
      className={"NumericField"}
      onValueChange={onChange}
      step={step}
      smallStep={step}
      min={min}
      max={max}
      type={"number"}
    >
      <NumberField.ScrubArea className="NumericFieldScrubArea">
        {label && (
          <label htmlFor={id} className="NumericFieldLabel">
            {label}
          </label>
        )}

        <NumberField.ScrubAreaCursor className="NumericFieldScrubAreaCursor">
          <CursorGrowIcon />
        </NumberField.ScrubAreaCursor>
      </NumberField.ScrubArea>

      <NumberField.Group className="NumericFieldGroup">
        <NumberField.Decrement className="NumericFieldDecrement">
          <MinusIcon />
        </NumberField.Decrement>
        <NumberField.Input
          className="NumericFieldInput"
          onInput={(e) => onChange(e.target.value)}
        />
        <NumberField.Increment className="NumericFieldIncrement">
          <PlusIcon />
        </NumberField.Increment>
      </NumberField.Group>
    </NumberField.Root>
  );
}

function CursorGrowIcon(props) {
  return (
    <svg
      width="26"
      height="14"
      viewBox="0 0 24 14"
      fill="black"
      stroke="white"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M19.5 5.5L6.49737 5.51844V2L1 6.9999L6.5 12L6.49737 8.5L19.5 8.5V12L25 6.9999L19.5 2V5.5Z" />
    </svg>
  );
}

function PlusIcon(props) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 10 10"
      fill="none"
      stroke="currentcolor"
      strokeWidth="1.6"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M0 5H5M10 5H5M5 5V0M5 5V10" />
    </svg>
  );
}

function MinusIcon(props) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 10 10"
      fill="none"
      stroke="currentcolor"
      strokeWidth="1.6"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M0 5H10" />
    </svg>
  );
}
