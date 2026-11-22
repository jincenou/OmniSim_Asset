"""Original native32999 updates with the validated cohort and storage adapter."""
import grid_context
import train_grid

train_grid.ROOT = grid_context.D
train_grid.BASE = grid_context.D / "data"

if __name__ == "__main__":
    train_grid.main()
