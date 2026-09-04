import { expect, it } from "vitest";
import { CleanupBag } from "./lifecycle";

it("runs cleanup once in reverse order", () => {
  const calls: number[] = [];
  const bag = new CleanupBag();
  bag.add(() => calls.push(1));
  bag.add(() => calls.push(2));
  bag.close();
  bag.close();
  bag.add(() => calls.push(3));
  expect(calls).toEqual([2, 1, 3]);
});
