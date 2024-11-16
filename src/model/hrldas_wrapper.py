# src/model/hrldas_wrapper.py
import os
import subprocess
import numpy as np
import xarray as xr
from pathlib import Path
import f90nml
import logging
from typing import Dict, Any

class HRLDASWrapper:
    """HRLDAS模型包装器"""

    def __init__(self,
                 hrldas_dir: Path,
                 forcing_dir: Path,
                 namelist_template: Path,
                 mptable: Path,
                 debug: bool = False):
        """
        初始化HRLDAS包装器

        Args:
            hrldas_dir: HRLDAS主目录
            forcing_dir: 强迫数据目录
            namelist_template: namelist模板
            mptable: MPTABLE.TBL文件
            debug: 是否开启调试模式
        """
        self.hrldas_dir = Path(hrldas_dir)
        self.forcing_dir = Path(forcing_dir)
        self.namelist_template = Path(namelist_template)
        self.mptable = Path(mptable)
        self.debug = debug

        # 验证路径
        self._validate_paths()

        # 读取模板
        self.namelist = f90nml.read(self.namelist_template)

        # 设置日志
        self.logger = logging.getLogger('HRLDAS')

    def run(self, params: Dict[str, float], work_dir: Path) -> Dict[str, np.ndarray]:
        """运行模型"""
        try:
            # 准备工作目录
            self.prepare_workspace(work_dir, params)

            # 切换到工作目录
            original_dir = os.getcwd()
            os.chdir(work_dir)

            # 运行模型
            cmd = str(self.hrldas_dir / "run" / "hrldas.exe")
            result = subprocess.run(
                [cmd],
                check=True,
                capture_output=True,
                text=True
            )

            # 检查输出
            if "ERROR" in result.stdout or "ERROR" in result.stderr:
                raise RuntimeError(f"Model run failed: {result.stderr}")

            # 读取输出
            output = self._read_output(work_dir)

            # 清理
            if not self.debug:
                self._cleanup(work_dir)

            return output

        except Exception as e:
            self.logger.error(f"Model run failed: {e}")
            raise

        finally:
            # 恢复目录
            os.chdir(original_dir)

    # ... [其他HRLDASWrapper类方法] ...

# src/model/parameter_handler.py
class NoahMPParamHandler:
    """Noah-MP参数处理器"""

    def __init__(self,
                 template_namelist: Path,
                 template_soilparm: Path,
                 template_vegparm: Path,
                 template_genparm: Path):
        """
        初始化参数处理器

        Args:
            template_namelist: namelist模板文件
            template_soilparm: SOILPARM.TBL模板
            template_vegparm: VEGPARM.TBL模板
            template_genparm: GENPARM.TBL模板
        """
        self.template_namelist = template_namelist
        self.template_soilparm = template_soilparm
        self.template_vegparm = template_vegparm
        self.template_genparm = template_genparm

        # 读取模板
        self.namelist_template = f90nml.read(template_namelist)
        self.soilparm_template = self._read_tbl(template_soilparm)
        self.vegparm_template = self._read_tbl(template_vegparm)
        self.genparm_template = self._read_tbl(template_genparm)

        # 参数映射
        self.param_mapping = {
            'SMCMAX': ('soilparm', 'maxsmc'),
            'BEXP': ('soilparm', 'bb'),
            'PSISAT': ('soilparm', 'satpsi'),
            'QUARTZ': ('soilparm', 'qtz'),
            'DKSAT': ('soilparm', 'satdk'),
            'THERIN': ('genparm', 'therin'),
            'F1': ('genparm', 'f1'),
            'SMCREF': ('soilparm', 'refsmc'),
            'SMCWLT': ('soilparm', 'wltsmc'),
            'SMCDRY': ('soilparm', 'drysmc')
        }

    def update_parameters(self, params: Dict[str, float], output_dir: Path):
        """更新参数文件"""
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 复制并更新模板
        namelist_data = self.namelist_template.copy()
        soilparm_data = self.soilparm_template.copy()
        vegparm_data = self.vegparm_template.copy()
        genparm_data = self.genparm_template.copy()

        # 更新参数
        for param_name, value in params.items():
            if param_name in self.param_mapping:
                file_type, param_key = self.param_mapping[param_name]

                if file_type == 'soilparm':
                    self._update_tbl_param(soilparm_data, param_key, value)
                elif file_type == 'vegparm':
                    self._update_tbl_param(vegparm_data, param_key, value)
                elif file_type == 'genparm':
                    self._update_tbl_param(genparm_data, param_key, value)
                else:  # namelist
                    self._update_namelist_param(namelist_data, param_key, value)

        # 写入更新后的文件
        f90nml.write(namelist_data, output_dir / 'namelist.input')
        self._write_tbl(soilparm_data, output_dir / 'SOILPARM.TBL')
        self._write_tbl(vegparm_data, output_dir / 'VEGPARM.TBL')
        self._write_tbl(genparm_data, output_dir / 'GENPARM.TBL')

    # ... [其他NoahMPParamHandler类方法] ...
