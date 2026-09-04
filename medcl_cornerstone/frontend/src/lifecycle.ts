export type Cleanup = () => void;

export class CleanupBag {
  private tasks: Cleanup[] = [];
  private closed = false;

  add(task: Cleanup): void {
    if (this.closed) task();
    else this.tasks.push(task);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const task of this.tasks.reverse()) {
      try { task(); } catch { /* cleanup is best effort */ }
    }
    this.tasks = [];
  }
}
