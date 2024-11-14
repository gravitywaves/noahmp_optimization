# noahmp_optimization

A parameter optimization system for the Noah-MP land surface model using the SCE-UA algorithm.

## Structure

- `config/`: Configuration files
- `src/`: Source code
- `templates/`: LaTeX templates
- `scripts/`: Utility scripts
- `tests/`: Test suite
- `docs/`: Documentation
- `data/`: Data directory
- `output/`: Output directory

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Configure the system in `config/`
2. Run optimization:
   ```bash
   python scripts/run_optimization.py
   ```
3. Generate report:
   ```bash
   python scripts/generate_report.py
   ```

## License

MIT License
