"""
Command Line Interface for Training Data Bot.

This module provides a comprehensive CLI using Typer for all bot operations.
"""

import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from ..core import settings, TaskType, ExportFormat, DocumentType
from ..bot import TrainingDataBot

# Initialize CLI app
app = typer.Typer(
    name="tdb",
    help="🧠 Training Data Curation Bot - Enterprise-grade training data curation for LLM fine-tuning",
    add_completion=False,
)

console = Console()


@app.command("process")
def process_documents(
    source_dir: Path = typer.Option(
        ...,
        "--source-dir", "-s",
        help="Directory containing source documents",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"),
        "--output-dir", "-o", 
        help="Output directory for generated datasets"
    ),
    task_types: Optional[List[str]] = typer.Option(
        None,
        "--task-type", "-t",
        help="Task types to execute (qa_generation, classification, summarization)"
    ),
    format: str = typer.Option(
        "jsonl",
        "--format", "-f",
        help="Export format (jsonl, csv, parquet, huggingface)"
    ),
    quality_filter: bool = typer.Option(
        True,
        "--quality-filter/--no-quality-filter",
        help="Apply quality filtering"
    ),
    max_workers: int = typer.Option(
        settings.processing.max_workers,
        "--max-workers", "-w",
        help="Maximum number of parallel workers"
    ),
):
    """Process documents and generate training datasets."""
    asyncio.run(_process_documents_async(
        source_dir, output_dir, task_types, format, quality_filter, max_workers
    ))


async def _process_documents_async(
    source_dir: Path,
    output_dir: Path, 
    task_types: Optional[List[str]],
    format: str,
    quality_filter: bool,
    max_workers: int
):
    """Async implementation of document processing."""
    
    # Validate task types
    if task_types:
        try:
            task_type_enums = [TaskType(t) for t in task_types]
        except ValueError as e:
            console.print(f"[red]Error: Invalid task type. {e}[/red]")
            raise typer.Exit(1)
    else:
        task_type_enums = [TaskType.QA_GENERATION, TaskType.CLASSIFICATION]
    
    # Validate export format
    try:
        export_format = ExportFormat(format)
    except ValueError:
        console.print(f"[red]Error: Invalid export format '{format}'. "
                     f"Valid options: {', '.join([f.value for f in ExportFormat])}[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize bot
        task_init = progress.add_task("Initializing Training Data Bot...", total=None)
        
        async with TrainingDataBot() as bot:
            progress.update(task_init, description="✅ Bot initialized")
            
            # Load documents
            task_load = progress.add_task("Loading documents...", total=None)
            try:
                documents = await bot.load_documents([source_dir])
                progress.update(task_load, description=f"✅ Loaded {len(documents)} documents")
            except Exception as e:
                progress.update(task_load, description=f"❌ Failed to load documents: {e}")
                raise typer.Exit(1)
            
            # Process documents
            task_process = progress.add_task("Processing documents...", total=None)
            try:
                dataset = await bot.process_documents(
                    documents=documents,
                    task_types=task_type_enums,
                    quality_filter=quality_filter,
                )
                progress.update(task_process, description=f"✅ Generated {len(dataset.examples)} examples")
            except Exception as e:
                progress.update(task_process, description=f"❌ Processing failed: {e}")
                raise typer.Exit(1)
            
            # Export dataset
            task_export = progress.add_task("Exporting dataset...", total=None)
            try:
                output_file = output_dir / f"dataset_{dataset.id}.{export_format.value}"
                exported_path = await bot.export_dataset(
                    dataset=dataset,
                    output_path=output_file,
                    format=export_format,
                )
                progress.update(task_export, description=f"✅ Exported to {exported_path}")
            except Exception as e:
                progress.update(task_export, description=f"❌ Export failed: {e}")
                raise typer.Exit(1)
    
    # Display results
    _display_results(dataset, exported_path)


@app.command("generate")
def generate_task_data(
    task_type: str = typer.Argument(
        ...,
        help="Task type (qa, classification, summarization, ner, red_teaming)"
    ),
    input_file: Path = typer.Option(
        ...,
        "--input-file", "-i",
        help="Input document file",
        exists=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file", "-o",
        help="Output dataset file"
    ),
    num_examples: int = typer.Option(
        100,
        "--num-examples", "-n",
        help="Maximum number of examples to generate"
    ),
):
    """Generate specific task data from a document."""
    asyncio.run(_generate_task_data_async(task_type, input_file, output_file, num_examples))


async def _generate_task_data_async(
    task_type: str,
    input_file: Path,
    output_file: Path,
    num_examples: int
):
    """Async implementation of task data generation."""
    
    # Map CLI task names to TaskType enums
    task_mapping = {
        "qa": TaskType.QA_GENERATION,
        "classification": TaskType.CLASSIFICATION,
        "summarization": TaskType.SUMMARIZATION,
        "ner": TaskType.NER,
        "red_teaming": TaskType.RED_TEAMING,
    }
    
    if task_type not in task_mapping:
        console.print(f"[red]Error: Unknown task type '{task_type}'. "
                     f"Valid options: {', '.join(task_mapping.keys())}[/red]")
        raise typer.Exit(1)
    
    task_type_enum = task_mapping[task_type]
    
    with console.status(f"[bold blue]Generating {task_type} data..."):
        async with TrainingDataBot() as bot:
            # Load single document
            documents = await bot.load_documents([input_file])
            
            # Process with specific task type
            dataset = await bot.process_documents(
                documents=documents,
                task_types=[task_type_enum],
            )
            
            # Limit examples if requested
            if len(dataset.examples) > num_examples:
                dataset.examples = dataset.examples[:num_examples]
                dataset.total_examples = len(dataset.examples)
            
            # Export
            await bot.export_dataset(
                dataset=dataset,
                output_path=output_file,
                format=ExportFormat.JSONL,
            )
    
    console.print(f"[green]✅ Generated {len(dataset.examples)} {task_type} examples[/green]")
    console.print(f"[blue]📁 Saved to: {output_file}[/blue]")


@app.command("evaluate")
def evaluate_dataset(
    dataset_file: Path = typer.Option(
        ...,
        "--dataset-file", "-d",
        help="Dataset file to evaluate",
        exists=True,
    ),
    output_report: Optional[Path] = typer.Option(
        None,
        "--output-report", "-o",
        help="Output HTML report file"
    ),
    detailed: bool = typer.Option(
        True,
        "--detailed/--summary",
        help="Generate detailed or summary report"
    ),
):
    """Evaluate quality of a dataset."""
    asyncio.run(_evaluate_dataset_async(dataset_file, output_report, detailed))


async def _evaluate_dataset_async(
    dataset_file: Path,
    output_report: Optional[Path],
    detailed: bool
):
    """Async implementation of dataset evaluation."""
    
    with console.status("[bold blue]Evaluating dataset quality..."):
        async with TrainingDataBot() as bot:
            # Load dataset (implementation depends on format)
            # For now, assume it's a stored dataset
            # In a real implementation, you'd load from the file
            
            console.print("[yellow]Note: Dataset loading from file not yet implemented[/yellow]")
            console.print("[blue]This would evaluate quality metrics like toxicity, bias, diversity[/blue]")


@app.command("dashboard")
def launch_dashboard(
    port: int = typer.Option(
        settings.dashboard.port,
        "--port", "-p",
        help="Port for dashboard server"
    ),
    host: str = typer.Option(
        settings.dashboard.host,
        "--host", "-h",
        help="Host for dashboard server"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode"
    ),
):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    
    console.print(f"[blue]🚀 Launching dashboard at http://{host}:{port}[/blue]")
    
    # Launch Streamlit app
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host,
    ]
    
    if not debug:
        cmd.extend(["--logger.level", "error"])
    
    subprocess.run(cmd)


@app.command("status")
def show_status():
    """Show bot status and statistics."""
    asyncio.run(_show_status_async())


async def _show_status_async():
    """Async implementation of status display."""
    
    async with TrainingDataBot() as bot:
        stats = bot.get_statistics()
        
        # Create status table
        table = Table(title="🧠 Training Data Bot Status")
        table.add_column("Component", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Details", style="green")
        
        # Documents
        table.add_row(
            "Documents",
            str(stats["documents"]["total"]),
            f"Size: {_format_bytes(stats['documents']['total_size'])}"
        )
        
        # Datasets  
        table.add_row(
            "Datasets",
            str(stats["datasets"]["total"]),
            f"Examples: {stats['datasets']['total_examples']}"
        )
        
        # Jobs
        active_jobs = stats["jobs"]["active"]
        table.add_row(
            "Active Jobs",
            str(active_jobs),
            f"Total: {stats['jobs']['total']}"
        )
        
        # Quality
        if stats["quality"]["total_examples"] > 0:
            approval_rate = (stats["quality"]["approved_examples"] / 
                           stats["quality"]["total_examples"]) * 100
            table.add_row(
                "Quality Approval",
                f"{approval_rate:.1f}%",
                f"{stats['quality']['approved_examples']}/{stats['quality']['total_examples']}"
            )
        
        console.print(table)


@app.command("templates")
def list_templates():
    """List available task templates."""
    
    table = Table(title="📋 Available Task Templates")
    table.add_column("Task Type", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")
    
    # Add built-in templates
    templates = [
        ("QA Generation", "Generate question-answer pairs", "max_questions, question_types"),
        ("Classification", "Classify text by intent/sentiment", "categories, confidence_threshold"), 
        ("Summarization", "Create text summaries", "max_length, style"),
        ("NER", "Extract named entities", "entity_types"),
        ("Red Teaming", "Generate adversarial examples", "attack_types"),
    ]
    
    for task_type, description, params in templates:
        table.add_row(task_type, description, params)
    
    console.print(table)


@app.command("config")
def show_config():
    """Show current configuration."""
    
    config_panel = Panel.fit(
        f"""
[bold cyan]Application Configuration[/bold cyan]

[yellow]Environment:[/yellow] {settings.environment}
[yellow]Debug Mode:[/yellow] {settings.debug}
[yellow]Log Level:[/yellow] {settings.log_level}

[bold blue]Processing[/bold blue]
[yellow]Max Workers:[/yellow] {settings.processing.max_workers}
[yellow]Chunk Size:[/yellow] {settings.processing.chunk_size}
[yellow]Batch Size:[/yellow] {settings.processing.batch_size}

[bold blue]Quality Thresholds[/bold blue]
[yellow]Toxicity:[/yellow] {settings.quality.toxicity_threshold}
[yellow]Bias:[/yellow] {settings.quality.bias_threshold}
[yellow]Similarity:[/yellow] {settings.quality.similarity_threshold}

[bold blue]Storage[/bold blue]
[yellow]Data Dir:[/yellow] {settings.storage.data_dir}
[yellow]Output Dir:[/yellow] {settings.storage.output_dir}
[yellow]Database:[/yellow] {settings.storage.database_url}
        """,
        title="⚙️ Configuration",
        border_style="blue"
    )
    
    console.print(config_panel)


def _display_results(dataset, output_path: Path):
    """Display processing results."""
    
    panel = Panel.fit(
        f"""
[bold green]✅ Processing Complete![/bold green]

[bold blue]Dataset Information:[/bold blue]
[yellow]• Total Examples:[/yellow] {len(dataset.examples)}
[yellow]• Quality Approved:[/yellow] {len([ex for ex in dataset.examples if ex.quality_approved])}
[yellow]• Dataset ID:[/yellow] {dataset.id}

[bold blue]Export Information:[/bold blue]
[yellow]• Output File:[/yellow] {output_path}
[yellow]• Format:[/yellow] {dataset.export_format.value}

[bold blue]Task Type Breakdown:[/bold blue]
{_format_task_breakdown(dataset)}
        """,
        title="🎉 Results",
        border_style="green"
    )
    
    console.print(panel)


def _format_task_breakdown(dataset) -> str:
    """Format task type breakdown for display."""
    breakdown = {}
    for example in dataset.examples:
        task_type = example.task_type.value if hasattr(example.task_type, 'value') else example.task_type
        breakdown[task_type] = breakdown.get(task_type, 0) + 1
    
    lines = []
    for task_type, count in breakdown.items():
        lines.append(f"[yellow]• {task_type.title()}:[/yellow] {count}")
    
    return "\n".join(lines)


def _format_bytes(bytes_size: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


if __name__ == "__main__":
    app() 