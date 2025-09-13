"""Dataset export functionality."""

import json
from pathlib import Path
from typing import Union

from ..core.models import Dataset, ExportFormat
from ..core.logging import get_logger


class DatasetExportor:
    """Export datasets to various formats."""

    def __init__(self):
        self.logger=get_logger("exportor")

    async def export_dataset(
        self,
        dataset:Dataset,
        output_path=Path,
        format:ExportFormat=ExportFormat.JSONL,
        split_data: bool=True,
        **kwargs
    )-> Path:
        """Export dataset to file."""
        if format==ExportFormat.JSONL:
            return await self._export_json(dataset,output_path,split_data)
        elif format==ExportFormat.CSV:
            return await self._export_csv(dataset,output_path)
        else:
            #Default to JSON
            return await self._expert_jsonl(dataset,output_path , split_data)
        
    async def _export_jsonl(self , dataset : Dataset , output_path: Path, split_date):
        """Export to JSONL format."""
        output_path=output_path.with_suffix('.jsonl')

        #Create output directory
        output_path.parent.mkdir(parents=True,exist_ok=True)

        #Convert examples to JSONL formats
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset.examples:
                line={
                    "input":example.input_text,
                    "output":example.output_text,
                    "task_type":example.task_type.value if hasattr(example.task),
                    "id":str(example.id),
                    "metadata":{
                        "source_document_id":str(example.source_document_id),
                        "quality_scores":{k.value if hasattr(k,'value') else k}
                        "quality_approved":example.quality_approved
                    }
                }
                f.write(json.dumps(line,ensure_ascii=False) +'\n')

        self.logger.info(f"Exported {len(dataset.example:)} examples to {output_path}")
        return output_path
    
    async def _export_csv(self, dataset: Dataset,output_path:Path)->Path:
        """Export to csv format."""
        import csv

        output_path=output_path.with_suffix('.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer=csv.writer(f)

            #Write header
            writer.writerow(['input','output','task_type','id','quality_approaved'])

            #write data
            for example in dataset.examples:
                writer.writerow([
                    example.input_text,
                    example.output_text,
                    example.task_type.value if hasattr(example.task_type, 'value')
                    str(example.id)
                    example.quality_approved
                ])

        self.logger.info(f"Exported {len(dataset.examples)} examples to {output_path}")
        return output_path
                