"""
최상위 폴더 하위의 모든 CIF 구조를 단일 기준 구조에 superimpose.

- 하위 디렉토리를 재귀적으로 탐색하여 모든 CIF 파일 수집
- 임의의 한 구조(기본: 알파벳 순 첫 번째 파일)를 reference로 사용
- 지정한 chain의 Cα 원자 기준으로 모든 파일을 superimpose
- 잔기 번호(residue sequence number) 기준으로 공통 Cα만 사용 (길이 상이 설계 허용)
- 입력 디렉토리 구조를 유지하며 output_root에 저장
- 원본 CIF 메타데이터(pLDDT, _ma_qa_metric_local 등) 완전 보존 (gemmi 사용)

Usage:
    python superimpose_all_by_chain.py \\
        --input_root INPUT_ROOT \\
        --output_root OUTPUT_ROOT \\
        [--chain A] \\
        [--reference FILE]
"""

import shutil
import argparse
import warnings
from pathlib import Path

import numpy as np
import gemmi
from Bio.PDB import MMCIFParser, Superimposer

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 구조 파싱 / Cα 추출
# ──────────────────────────────────────────────

def get_ca_dict(chain):
    """
    Chain에서 {residue_seq_num: CA_atom} dict 반환.
    삽입 코드가 있는 잔기 및 비표준 잔기(HETATM)는 제외.
    """
    return {
        res.id[1]: res["CA"]
        for res in chain.get_residues()
        if res.id[0] == " " and "CA" in res
    }


def get_matched_ca_pairs(ref_ca_dict, mob_chain):
    """
    Reference CA dict와 mobile chain에서
    공통 잔기 번호(residue seq num) 기준 Cα 원자쌍을 반환.
    CDR 길이가 달라도 공통 프레임워크 잔기로 alignment 가능.
    """
    mob_ca_dict = get_ca_dict(mob_chain)
    common = sorted(set(ref_ca_dict.keys()) & set(mob_ca_dict.keys()))
    return [ref_ca_dict[i] for i in common], [mob_ca_dict[i] for i in common]


# ──────────────────────────────────────────────
# 좌표 변환 및 CIF 저장 (gemmi)
# ──────────────────────────────────────────────

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
    원본 CIF 파일의 모든 메타데이터를 유지하면서
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


# ──────────────────────────────────────────────
# 메인 로직
# ──────────────────────────────────────────────

def superimpose_all(
    input_root: str,
    output_root: str,
    chain_id: str = "A",
    reference_path: str = None,
):
    """
    input_root 하위의 모든 CIF 파일을 단일 reference 기준으로 superimpose.

    Parameters
    ----------
    input_root     : CIF 파일을 포함하는 최상위 디렉토리 (재귀 탐색)
    output_root    : 결과 저장 디렉토리 (입력 구조 미러링)
    chain_id       : Superimpose 기준 chain ID (대문자로 자동 변환)
    reference_path : 기준 CIF 파일 경로. 미지정 시 input_root 내 첫 번째 파일 사용
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    chain_id = chain_id.upper()

    # ── 파일 수집 ──────────────────────────────
    all_files = sorted(input_root.rglob("*.cif"))
    if not all_files:
        print(f"[ERROR] CIF 파일을 찾을 수 없습니다: {input_root}")
        return

    total = len(all_files)
    print(f"총 CIF 파일 수  : {total:,}")
    print(f"Superimpose Chain: {chain_id}")

    # ── Reference 결정 ─────────────────────────
    if reference_path:
        ref_path = Path(reference_path).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference 파일 없음: {ref_path}")
    else:
        ref_path = all_files[0]

    try:
        print(f"Reference 구조  : {ref_path.relative_to(input_root)}")
    except ValueError:
        print(f"Reference 구조  : {ref_path}")

    # ── Reference 파싱 ─────────────────────────
    parser = MMCIFParser(QUIET=True)
    try:
        ref_struct = parser.get_structure("ref", str(ref_path))
    except Exception as e:
        raise RuntimeError(f"Reference 파싱 실패: {e}")

    ref_model = ref_struct[0]
    if chain_id not in ref_model:
        available = [c.id for c in ref_model.get_chains()]
        raise ValueError(
            f"Reference에 Chain '{chain_id}' 없음. "
            f"사용 가능한 chain: {available}"
        )

    ref_ca_dict = get_ca_dict(ref_model[chain_id])
    if not ref_ca_dict:
        raise ValueError(f"Reference Chain '{chain_id}'에 Cα 원자 없음")
    print(f"Reference Cα 수 : {len(ref_ca_dict)}")
    print()

    # ── Reference 복사 ─────────────────────────
    ref_out = output_root / ref_path.relative_to(input_root)
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(ref_path), str(ref_out))

    # ── Superimpose ────────────────────────────
    sup = Superimposer()
    success = skip = error = 0

    for i, cif_path in enumerate(all_files, 1):
        if cif_path == ref_path:
            continue

        out_path = output_root / cif_path.relative_to(input_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            mob_struct = parser.get_structure("mob", str(cif_path))
            mob_model = mob_struct[0]

            if chain_id not in mob_model:
                print(f"  [WARN] Chain '{chain_id}' 없음: {cif_path.relative_to(input_root)}")
                skip += 1
                continue

            ref_ca, mob_ca = get_matched_ca_pairs(ref_ca_dict, mob_model[chain_id])

            if len(ref_ca) < 3:
                print(
                    f"  [WARN] 공통 Cα {len(ref_ca)}개 (최소 3 필요): "
                    f"{cif_path.relative_to(input_root)}"
                )
                skip += 1
                continue

            sup.set_atoms(ref_ca, mob_ca)
            rot, tran = sup.rotran

            apply_transform_to_cif(cif_path, out_path, rot, tran)
            success += 1

        except Exception as e:
            print(f"  [ERROR] {cif_path.relative_to(input_root)}: {e}")
            error += 1

        if i % 1000 == 0 or i == total:
            print(f"  진행: {i:,}/{total:,}  (성공 {success:,} / 스킵 {skip:,} / 오류 {error:,})")

    print(f"\n완료: 성공 {success:,}개, 스킵 {skip:,}개, 오류 {error:,}개")
    print(f"결과 저장 위치: {output_root}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="최상위 폴더 하위의 모든 CIF 구조를 단일 reference 기준으로 superimpose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_root",
        required=True,
        metavar="DIR",
        help="CIF 파일이 있는 최상위 디렉토리 (하위 폴더 포함 재귀 탐색)",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        metavar="DIR",
        help="결과를 저장할 최상위 디렉토리 (입력 디렉토리 구조 미러링)",
    )
    parser.add_argument(
        "--chain",
        default="A",
        metavar="CHAIN_ID",
        help="Superimpose 기준 chain ID (예: A, B)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="FILE",
        help="기준 CIF 파일 경로 (미지정 시 input_root 내 알파벳 순 첫 번째 파일 자동 선택)",
    )
    args = parser.parse_args()

    superimpose_all(
        input_root=args.input_root,
        output_root=args.output_root,
        chain_id=args.chain,
        reference_path=args.reference,
    )


if __name__ == "__main__":
    main()
