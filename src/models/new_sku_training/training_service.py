"""Patch-only PatchCore training workers."""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from PyQt5.QtCore import QThread, pyqtSignal

class _PatchTrainingWorkerBase(QThread):
    statusSignal=pyqtSignal(str); finishedSignal=pyqtSignal(dict); errorSignal=pyqtSignal(str)
    def __init__(self, config:Dict[str,Any], project_root:str, parent=None):
        super().__init__(parent); self.config=dict(config); self.project_root=Path(project_root).resolve(); self._process:Optional[subprocess.Popen]=None
    def stop(self):
        if self._process is not None and self._process.poll() is None:
            try:self._process.terminate()
            except Exception:pass
    def _run_cycle(self,payload:Dict[str,Any],config_path:Path):
        config_path.parent.mkdir(parents=True,exist_ok=True); config_path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        script=self.project_root/'src/models/new_sku_training/patch_only_pipeline/main_patch_training_cycle.py'
        cmd=[sys.executable,str(script),'--config',str(config_path),'--workers',str(max(1,int(payload.get('max_parallel_workers',1))))]
        env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['PYTHONUNBUFFERED']='1'
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' and hasattr(subprocess,'CREATE_NO_WINDOW') else 0
        self._process=subprocess.Popen(cmd,cwd=str(script.parent),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,env=env,creationflags=flags)
        recent=[]
        assert self._process.stdout is not None
        for raw in self._process.stdout:
            line=raw.rstrip('\r\n')
            if line:
                recent.append(line); recent=recent[-80:]; self.statusSignal.emit(line)
        code=self._process.wait(); self._process=None
        if code!=0: raise RuntimeError(f'Patch-only training exited with code {code}.\n\n'+ '\n'.join(recent[-16:]))
        summary=Path(payload['output_root'])/'main_patch_training_summary.json'
        if not summary.is_file(): raise RuntimeError(f'Training summary not found: {summary}')
        result=json.loads(summary.read_text(encoding='utf-8')); result['summary_path']=str(summary); result['cycle_output_root']=str(summary.parent); result['completed_at']=datetime.now().isoformat(timespec='seconds'); return result

class LocalTrainingWorker(_PatchTrainingWorkerBase):
    def run(self):
        try:
            model=Path(str(self.config['out_path'])).resolve(); role=str(self.config['role']); root=model.parent
            payload={'cycle_name':f'{role}_patch_only_training','output_root':str(root),'max_parallel_workers':1,'cuda_visible_devices':'0','cpu_threads_per_worker':1,'device':'auto','image_batch_size':int(self.config.get('batch_size',32)),'num_workers':int(self.config.get('num_workers',0)),'input_size':224,'feature_patch_size':3,'coreset_percentage':float(self.config.get('coreset_percentage',0.1)),'seed':0,'recursive':True,'jobs':[{'name':role,'enabled':True,'patch_folder':str(self.config['patch_folder']),'out_model':str(model)}]}
            summary=self._run_cycle(payload,root/'patch_training_run_config.json'); item=(summary.get('results') or [])[0]
            if item.get('status')!='success': raise RuntimeError(str(item.get('error','Training failed')))
            result={'sku_name':self.config.get('sku_name'),'role':role,'display_name':self.config.get('display_name',role),'pipeline':'patch_only','model_path':item.get('out_model_path',str(model)),'summary_path':summary.get('summary_path',''),'worker_log':item.get('worker_log',''),'patch_folder':self.config['patch_folder'],'generated_training_patch_count':int(item.get('patch_image_count',0) or 0),'successful_input_count':int(item.get('successful_patch_image_count',0) or 0),'failed_input_count':int(item.get('failed_patch_image_count',0) or 0),'memory_bank_shape':[int(item.get('memory_bank_patch_count',0) or 0),int(item.get('memory_bank_feature_dimension',0) or 0)],'total_pipeline_time':float(item.get('elapsed_seconds',0) or 0)}
            self.finishedSignal.emit(result)
        except Exception as exc:self._process=None; self.errorSignal.emit(f'{type(exc).__name__}: {exc}')

class FiveSideTrainingWorker(_PatchTrainingWorkerBase):
    def run(self):
        try:
            payload=dict(self.config); root=Path(str(payload['cycle_config_root'])).resolve(); payload.pop('cycle_config_root',None)
            result=self._run_cycle(payload,root/'main_patch_training_config.json'); self.finishedSignal.emit(result)
        except Exception as exc:self._process=None; self.errorSignal.emit(f'{type(exc).__name__}: {exc}')
