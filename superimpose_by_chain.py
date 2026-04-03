"""
Superimpose CIF structures by a specified chain (Cα atoms).
Groups files by design ID and aligns model_1~N onto the reference model.
All chains move together as a rigid body.
Original CIF metadata (pLDDT, _ma_qa_metric_local, etc.) is fully preserved.

Usage:
    python superimpose_by_chain.py [--input_dir DIR] [--output_dir DIR]
                                   [--chain A] [--reference_model 0]
"""

import re
import time
import shutil
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import gemmi
from Bio.PDB import MMCIFParser, Superimposer

warnings.filterwarnings("ignore")


def parse_structure_with_retry(parser, name, filepath, retries=5, delay=10):
    """타임아웃 시 재시도하며 CIF 구조를 파싱."""
    for attempt in range(retries):
        try:
            return parser.get_structure(name, str(filepath))
        except (TimeoutError, OSError) as e:
            if attempt < retries - 1:
                print(f"    [재시도 {attempt+1}/{retries}] {filepath.name}: {e}")
                time.sleep(delay)
            else:
                raise


def get_ca_atoms(chain):
    """Chain에서 Cα 원자 리스트를 반환 (standard residue only)."""
    ca_atoms = []
    for residue in chain.get_residues():
        if residue.id[0] == " " and "CA" in residue:
            ca_atoms.append(residue["CA"])
    return ca_atoms


def _ensure_entity_categories(block):
    """
    Mol*가 단백질로 인식하기 위한 필수 mmCIF 카테고리가 없으면 추가.

    _entity, _entity_poly, _struct_asym이 없으면 _atom_site에서
    chain/entity 정보를 추출하여 생성한다.
    Mol*는 이 카테고리가 없으면 cartoon 표현을 제공하지 않는다.
    """
    if block.find_value("_entity.id") not in ("", None):
        return
    entity_loop = block.find(["_entity.id"])
    if entity_loop and len(entity_loop) > 0:
        return

    site_table = block.find(
        "_atom_site.",
        ["label_entity_id", "label_asym_id", "label_comp_id"],
    )
    if not site_table:
        return

    entities = {}
    residue_names = {}
    for row in site_table:
        eid = row[0] if row[0] not in (".", "?", "") else "1"
        asym = row[1] if row[1] not in (".", "?", "") else "A"
        comp = row[2]
        entities.setdefault(eid, set()).add(asym)
        residue_names.setdefault(eid, set()).add(comp)

    entity_lp = block.init_loop("_entity.", ["id", "type", "pdbx_description"])
    for eid in sorted(entities):
        entity_lp.add_row([eid, "polymer", "polymer"])

    poly_lp = block.init_loop(
        "_entity_poly.", ["entity_id", "type", "pdbx_strand_id"]
    )
    for eid, asym_ids in sorted(entities.items()):
        strand = ",".join(sorted(asym_ids))
        poly_lp.add_row([eid, "polypeptide(L)", strand])

    asym_lp = block.init_loop("_struct_asym.", ["id", "entity_id"])
    for eid, asym_ids in sorted(entities.items()):
        for asym_id in sorted(asym_ids):
            asym_lp.add_row([asym_id, eid])


def apply_transform_to_cif(input_path, output_path, rot, tran):
    """
    원본 CIF 파일의 모든 메타데이터(pLDDT, _ma_qa_metric_local 등)를 유지하면서
    rotation/translation을 좌표(_atom_site.Cartn_x/y/z)에만 적용하여 저장.

    Mol*에서 cartoon 표현이 가능하도록 필수 entity 카테고리도 보충한다.

    변환식: new_coord = old_coord @ rot + tran  (BioPython Superimposer 규약)
    """
    doc = gemmi.cif.read(str(input_path))
    block = doc.sole_block()

    table = block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
    for row in table:
        coord = np.array([float(row[0]), float(row[1]), float(row[2])])
        new_coord = coord @ rot + tran
        row[0] = f"{new_coord[0]:.3f}"
        row[1] = f"{new_coord[1]:.3f}"
        row[2] = f"{new_coord[2]:.3f}"

    _ensure_entity_categories(block)

    doc.write_file(str(output_path))


def superimpose_models(
    input_dir: str,
    output_dir: str,
    chain_id: str = "A",
    reference_model_idx: int = 0,
):
    """
    input_dir 내 CIF 파일을 디자인 ID별로 그룹화하고,
    각 그룹에서 reference_model을 기준으로 나머지 모델을 지정 Chain의 Cα로 superimpose.
    원본 CIF의 모든 메타데이터(pLDDT 등)를 유지하며 결과를 output_dir에 저장.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chain_id = chain_id.upper()

    # 파일 그룹화: design_id -> {model_idx: filepath}
    pattern = re.compile(r"^(.+)_model_(\d+)\.cif$")
    groups = defaultdict(dict)

    for fname in sorted(input_dir.glob("*.cif")):
        m = pattern.match(fname.name)
        if m:
            design_id = m.group(1)
            model_idx = int(m.group(2))
            groups[design_id][model_idx] = fname

    print(f"총 디자인 그룹 수: {len(groups)}")
    print(f"Superimpose 기준 Chain: {chain_id}")
    print(f"Reference 모델 인덱스: {reference_model_idx}")

    parser = MMCIFParser(QUIET=True)
    sup = Superimposer()

    success_count = 0
    error_count = 0

    for design_id, model_files in sorted(groups.items()):
        if reference_model_idx not in model_files:
            print(f"  [WARN] {design_id}: reference model_{reference_model_idx} 없음, skip")
            error_count += 1
            continue

        # Reference 구조 로드
        ref_path = model_files[reference_model_idx]
        try:
            ref_struct = parse_structure_with_retry(parser, "ref", ref_path)
        except (TimeoutError, OSError) as e:
            print(f"  [ERROR] {design_id} reference: {e}, skip")
            error_count += 1
            continue
        ref_model = ref_struct[0]

        if chain_id not in ref_model:
            print(f"  [WARN] {design_id}: reference에 Chain {chain_id} 없음, skip")
            error_count += 1
            continue

        ref_ca = get_ca_atoms(ref_model[chain_id])
        if not ref_ca:
            print(f"  [WARN] {design_id}: reference Chain {chain_id}에 Cα 없음, skip")
            error_count += 1
            continue

        # Reference 모델은 변환 없이 원본 그대로 복사
        shutil.copy2(str(ref_path), str(output_dir / ref_path.name))

        # 나머지 모델 superimpose
        for idx, mob_path in sorted(model_files.items()):
            if idx == reference_model_idx:
                continue

            try:
                mob_struct = parse_structure_with_retry(parser, "mob", mob_path)
            except (TimeoutError, OSError) as e:
                print(f"  [ERROR] {design_id} model_{idx}: {e}, skip")
                error_count += 1
                continue
            mob_model = mob_struct[0]

            if chain_id not in mob_model:
                print(f"  [WARN] {design_id} model_{idx}: Chain {chain_id} 없음, skip")
                error_count += 1
                continue

            mob_ca = get_ca_atoms(mob_model[chain_id])

            if len(mob_ca) != len(ref_ca):
                print(
                    f"  [WARN] {design_id} model_{idx}: "
                    f"Cα 수 불일치 (ref={len(ref_ca)}, mob={len(mob_ca)}), skip"
                )
                error_count += 1
                continue

            # 지정 Chain Cα 기준으로 superimpose 계산 → rotation/translation 획득
            sup.set_atoms(ref_ca, mob_ca)
            rot, tran = sup.rotran

            # 원본 CIF 메타데이터를 유지하면서 좌표만 변환하여 저장
            out_path = output_dir / mob_path.name
            try:
                apply_transform_to_cif(mob_path, out_path, rot, tran)
            except Exception as e:
                print(f"  [ERROR] {design_id} model_{idx} 저장 실패: {e}, skip")
                error_count += 1
                continue

            success_count += 1

        # 진행 상황 출력 (500개마다)
        total_done = success_count + error_count
        if total_done % 500 == 0 and total_done > 0:
            print(f"  진행: {total_done}개 처리 완료")

    print(f"\n완료: 성공 {success_count}개, 오류 {error_count}개")
    print(f"결과 저장 위치: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="지정 Chain 기준으로 CIF 구조 superimposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="입력 CIF 파일 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="출력 CIF 파일 디렉토리 (없으면 자동 생성)",
    )
    parser.add_argument(
        "--chain",
        default="A",
        metavar="CHAIN_ID",
        help="Superimpose 기준 chain ID (예: A, B)",
    )
    parser.add_argument(
        "--reference_model",
        type=int,
        default=0,
        metavar="N",
        help="기준 모델 인덱스",
    )
    args = parser.parse_args()

    superimpose_models(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chain_id=args.chain,
        reference_model_idx=args.reference_model,
    )


if __name__ == "__main__":
    main()
