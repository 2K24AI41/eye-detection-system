"""Tests for the metadata integrity checks of scripts/validate_labels.py (load_metadata).

Only synthetic files in temporary folders are used: nothing under the real dataset/ is read,
the frozen build_dedup_manifest.py is only READ AS TEXT (never imported or run), and no
window, webcam or annotation tool is involved.
"""
import ast
import contextlib
import csv
import hashlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_helpers as h  # noqa: E402

vl = h.vl
ValidationError = vl.ValidationError
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
HEADER = ["filename", "participant_id", "condition", "sub_condition", "capture_time", "notes"]
WHEN = "2026-01-01T10:00:00+05:00"


def row(name="normal_P01_001.png", pid="P01", cond="normal", sub="frontal", when=WHEN, notes=""):
    return [name, pid, cond, sub, when, notes]


def csv_text(rows, header=HEADER, eol="\n"):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator=eol)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


class MetadataCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()

    def write(self, text="", raw=None):
        data = raw if raw is not None else text.encode("utf-8")
        (self.dataset / "metadata.csv").write_bytes(data)

    def load(self):
        return vl.load_metadata(self.dataset)

    def expect_error(self, *fragments):
        with self.assertRaises(ValidationError) as caught:
            self.load()
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        self.assert_blind(message)
        return message

    def assert_blind(self, message):
        """No filename, filename stem, participant-style name or folder path."""
        self.assertIsNone(re.search(r"[\w.\-]+\.(png|jpe?g|bmp)", message, re.IGNORECASE), message)
        for hidden in (str(self.root), self.root.name, str(self.dataset), self.dataset.as_posix(),
                       "normal_P01", "bg_NONE", "dim_P01", "secret"):
            self.assertNotIn(hidden, message)


# ============================================================ valid input is unchanged
class ValidMetadataTests(MetadataCase):
    def test_every_valid_condition_sub_condition_combination_is_accepted(self):
        rows, n = [], 0
        for cond, subs in vl.CONDITION_SUBS.items():
            for sub in subs:
                for pid in (("NONE",) if cond == "bg" else ("P01", "P02")):
                    n += 1
                    rows.append(row(f"{cond}_{pid}_{n:03d}.png", pid, cond, sub))
        self.write(csv_text(rows))
        self.assertEqual(len(self.load()), len(rows))

    def test_the_documented_collection_groups_are_all_valid(self):
        groups = [("normal", "frontal"), ("dim", "dim"), ("strong", "side"), ("turn", "30deg"),
                  ("turn", "60deg"), ("tilt", "roll"), ("tilt", "down"), ("dist", "near"),
                  ("dist", "far"), ("bg", "background")]               # protocol section 11.3
        rows = [row(f"{c}_{'NONE' if c == 'bg' else 'P01'}_{i:03d}.png",
                    "NONE" if c == "bg" else "P01", c, s) for i, (c, s) in enumerate(groups, 1)]
        self.write(csv_text(rows))
        self.assertEqual(len(self.load()), 10)

    def test_return_value_is_exactly_what_a_plain_dictreader_gives(self):
        rows = [row("normal_P01_001.png", notes='a, "quoted" note'),
                row("normal_P01_002.png", sub="frontal", notes=""),
                row("bg_NONE_001.png", "NONE", "bg", "background", notes="wall")]
        text = csv_text(rows)
        self.write(text)
        with open(self.dataset / "metadata.csv", newline="", encoding="utf-8-sig") as handle:
            expected = {r["filename"]: r for r in csv.DictReader(handle)}
        got = self.load()
        self.assertEqual(got, expected)
        self.assertEqual(list(got), list(expected))                      # same order
        self.assertEqual(got["normal_P01_001.png"]["notes"], 'a, "quoted" note')

    def test_bom_lf_crlf_and_mixed_line_endings(self):
        rows = [row("normal_P01_001.png"), row("normal_P01_002.png")]
        for label, raw in (
                ("lf", csv_text(rows).encode()),
                ("crlf", csv_text(rows, eol="\r\n").encode()),
                ("bom+lf", b"\xef\xbb\xbf" + csv_text(rows).encode()),
                ("bom+crlf", b"\xef\xbb\xbf" + csv_text(rows, eol="\r\n").encode()),
                ("mixed", csv_text(rows[:1], eol="\r\n").encode() +
                 csv_text(rows[1:], eol="\n").encode().split(b"\n", 1)[1])):
            with self.subTest(label=label):
                self.write(raw=raw)
                self.assertEqual(sorted(self.load()), ["normal_P01_001.png", "normal_P01_002.png"])

    def test_quoted_notes_with_commas_quotes_and_line_breaks(self):
        notes = 'one, two\n"three"\r\nfour'
        self.write(csv_text([row(notes=notes), row("normal_P01_002.png")]))
        got = self.load()
        self.assertEqual(got["normal_P01_001.png"]["notes"], notes)
        self.assertEqual(len(got), 2)

    def test_empty_notes_and_a_bare_quote_inside_an_unquoted_field_are_accepted(self):
        self.write(raw=(",".join(HEADER) + "\n"
                        + "normal_P01_001.png,P01,normal,frontal," + WHEN + ',5" wall\n').encode())
        self.assertEqual(self.load()["normal_P01_001.png"]["notes"], '5" wall')

    def test_extra_header_columns_are_still_allowed_and_values_are_not_changed(self):
        text = csv_text([row() + ["extra value"]], header=HEADER + ["extra"])
        self.write(text)
        got = self.load()
        self.assertEqual(got["normal_P01_001.png"]["extra"], "extra value")

    def test_valid_metadata_passes_the_whole_validator_unchanged(self):
        dataset = h.make_dataset(self.root / "full", n_person=3, n_bg=1)
        h.annotate_everything(dataset)
        report = vl.validate_dataset(dataset, 1)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"]["annotated_images"], 4)
        self.assertEqual(vl.is_background(vl.load_metadata(dataset)["bg_NONE_001.png"]), True)


# ====================================================================== duplicates
class DuplicateTests(MetadataCase):
    def test_duplicate_filename_is_rejected_with_both_line_numbers(self):
        self.write(csv_text([row("normal_P01_001.png"), row("normal_P01_002.png"),
                             row("normal_P01_001.png", notes="again")]))
        message = self.expect_error("line 4", "repeats the one on line 2")
        self.assertNotIn("normal_P01_001", message)

    def test_an_identical_repeated_row_is_also_a_duplicate(self):
        self.write(csv_text([row(), row()]))
        self.expect_error("line 3", "repeats the one on line 2")

    def test_duplicate_filename_in_different_conditions_is_rejected(self):
        self.write(csv_text([row("x_P01_001.png"), row("x_P01_001.png", cond="dim", sub="dim")]))
        self.expect_error("repeats")

    def test_duplicate_column_names_are_rejected(self):
        header = HEADER + ["notes"]
        self.write(csv_text([row() + [""]], header=header))
        self.expect_error("duplicate column name(s) ['notes']")
        header = ["filename", "filename", "participant_id", "condition", "sub_condition",
                  "capture_time", "notes"]
        self.write(csv_text([], header=header))
        self.expect_error("duplicate column name")

    def test_the_whole_validator_reports_a_duplicate_as_one_clean_error(self):
        dataset = h.make_dataset(self.root / "full", n_person=2, n_bg=1)
        path = dataset / "metadata.csv"
        lines = path.read_text(encoding="utf-8").splitlines(True)
        path.write_text("".join(lines + [lines[1]]), encoding="utf-8", newline="")
        report = vl.validate_dataset(dataset, 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("repeats the one on line 2", report["errors"][0])
        self.assertNotIn("normal_P01_001", report["errors"][0])


# ================================================================ schema and rows
class SchemaAndRowTests(MetadataCase):
    def test_each_of_the_six_columns_is_required(self):
        for column in HEADER:
            with self.subTest(missing=column):
                keep = [c for c in HEADER if c != column]
                rows = [[v for c, v in zip(HEADER, row()) if c != column]]
                self.write(csv_text(rows, header=keep))
                self.expect_error("lacks the columns", column)

    def test_empty_file_bom_only_and_header_only(self):
        for label, raw in (("empty", b""), ("bom only", b"\xef\xbb\xbf"),
                           ("header only", csv_text([]).encode())):
            with self.subTest(label=label):
                self.write(raw=raw)
                self.expect_error("is empty or lacks the columns")

    def test_row_with_missing_values_names_the_line(self):
        self.write(csv_text([row("normal_P01_001.png"), ["normal_P01_002.png", "P01"]]))
        self.expect_error("line 3", "wrong number of values")

    def test_row_with_extra_values_names_the_line(self):
        self.write(csv_text([row() + ["surplus"]]))
        self.expect_error("line 2", "wrong number of values")

    def test_empty_and_whitespace_only_required_values_are_rejected(self):
        for index, column in enumerate(HEADER):
            for value in ("", " ", "   ", "\t"):
                with self.subTest(column=column, value=repr(value)):
                    values = row()
                    values[index] = value
                    self.write(csv_text([values]))
                    if column == "notes":
                        self.assertEqual(len(self.load()), 1)            # notes may be empty
                    else:
                        self.expect_error("line 2", f"{column} is empty")

    def test_error_names_the_line_of_the_bad_row_not_the_first(self):
        rows = [row("normal_P01_00%d.png" % i) for i in range(1, 5)]
        rows[2][2] = "zzz"
        self.write(csv_text(rows))
        self.expect_error("line 4", "condition is not one of")


# ================================================================ allowed values
class AllowedValueTests(MetadataCase):
    def test_unknown_conditions_are_rejected_without_echoing_the_value(self):
        for bad in ("zzz", "Normal", "NORMAL", " normal", "normal ", "normal ", "tilt/roll"):
            with self.subTest(condition=repr(bad)):
                self.write(csv_text([row(cond=bad)]))
                message = self.expect_error("line 2", "condition is not one of")
                if bad in ("zzz", "NORMAL", "tilt/roll"):          # not part of the allowed list
                    self.assertNotIn(bad, message)                  # the bad value is not echoed

    def test_every_invalid_condition_sub_condition_pairing_is_rejected(self):
        for cond, subs in vl.CONDITION_SUBS.items():
            for other_cond, other_subs in vl.CONDITION_SUBS.items():
                for sub in other_subs:
                    if sub in subs:
                        continue
                    with self.subTest(cond=cond, sub=sub):
                        pid = "NONE" if cond == "bg" else "P01"
                        self.write(csv_text([row("a_%s_001.png" % pid, pid, cond, sub)]))
                        self.expect_error("line 2", "sub_condition is not allowed for this condition")

    def test_unknown_and_case_changed_sub_conditions_are_rejected(self):
        for sub in ("nonsense", "Frontal", "frontal ", " frontal", ""):
            with self.subTest(sub=repr(sub)):
                self.write(csv_text([row(sub=sub)]))
                self.expect_error("line 2")

    def test_the_sub_condition_error_does_not_reveal_the_condition(self):
        self.write(csv_text([row("turn_P01_001.png", cond="turn", sub="near")]))
        message = self.expect_error("sub_condition is not allowed")
        for word in ("turn", "30deg", "60deg", "near"):
            self.assertNotIn(word, message)

    def test_valid_participants_and_background_consistency(self):
        self.write(csv_text([row("normal_P01_001.png", "P01"), row("normal_P02_002.png", "P02"),
                             row("bg_NONE_001.png", "NONE", "bg", "background")]))
        self.assertEqual(len(self.load()), 3)

    def test_none_is_only_permitted_with_bg(self):
        for cond, sub in (("normal", "frontal"), ("dim", "dim"), ("turn", "30deg"),
                          ("glasses", "glasses")):
            with self.subTest(cond=cond):
                self.write(csv_text([row("a_NONE_001.png", "NONE", cond, sub)]))
                self.expect_error("line 2", "participant_id does not match the condition type")

    def test_p01_and_p02_are_only_permitted_with_person_conditions(self):
        for pid in ("P01", "P02"):
            with self.subTest(pid=pid):
                self.write(csv_text([row("bg_x_001.png", pid, "bg", "background")]))
                self.expect_error("line 2", "participant_id does not match the condition type")

    def test_unknown_participants_are_rejected(self):
        for pid in ("P03", "p01", "P01 ", "none", "NONE ", "P1"):
            with self.subTest(pid=repr(pid)):
                self.write(csv_text([row(pid=pid)]))
                self.expect_error("line 2", "participant_id must be one of")


# ======================================================== encodings and parse errors
class ParsingAndFilesystemErrorTests(MetadataCase):
    def test_invalid_utf8_is_a_clean_validation_error(self):
        for label, raw in (
                ("in a value", (csv_text([]) + "bad\xff").encode("latin-1")),
                ("after a BOM", b"\xef\xbb\xbf" + csv_text([]).encode() + b"a\xff,P01,normal,frontal,t,\n"),
                ("in the header", b"file\xffname,participant_id\n")):
            with self.subTest(label=label):
                self.write(raw=raw)
                self.expect_error("not valid UTF-8")

    def test_nul_bytes_are_clean_validation_errors(self):
        def with_nul(text):                       # csv.writer refuses NUL, so insert it afterwards
            return text.replace("#", "\x00")
        for label, text in (("in a value", with_nul(csv_text([row(notes="a#b")]))),
                            ("in a filename", with_nul(csv_text([row("normal#_P01_001.png")]))),
                            ("in the header", csv_text([row()]).replace("notes", "no\x00tes", 1))):
            with self.subTest(label=label):
                self.write(raw=text.encode())
                self.expect_error()

    def test_nul_check_does_not_depend_on_the_csv_module(self):
        """Newer Python versions let NUL through the csv module, so the loader checks itself."""
        checked = {}
        vl._check_metadata_row(dict(zip(HEADER, row())), 2, checked)         # a clean row passes
        with self.assertRaises(ValidationError) as caught:
            vl._check_metadata_row(dict(zip(HEADER, row(notes="a\x00b"))), 7, {})
        self.assertIn("line 7", str(caught.exception))
        self.assertIn("NUL", str(caught.exception))

    def test_malformed_quoting_is_a_clean_validation_error(self):
        for label, raw in (
                ("unbalanced quote", (",".join(HEADER) + '\nnormal_P01_001.png,P01,normal,frontal,'
                                      + WHEN + ',"abc\nnormal_P01_002.png,P01,normal,frontal,t,\n').encode()),
                ("text after closing quote", (",".join(HEADER) + '\nnormal_P01_001.png,P01,normal,frontal,'
                                              + WHEN + ',"ab"cd\n').encode())):
            with self.subTest(label=label):
                self.write(raw=raw)
                self.expect_error("malformed CSV")

    def test_a_missing_metadata_file_is_clean_and_has_no_path(self):
        with self.assertRaises(ValidationError) as caught:
            vl.load_metadata(self.root / "no_such_folder")
        message = str(caught.exception)
        self.assertTrue(message.startswith("Cannot read metadata.csv"))
        self.assert_blind(message)

    def test_metadata_that_is_a_folder_is_clean_and_has_no_path(self):
        (self.dataset / "metadata.csv").mkdir()
        self.expect_error("Cannot read metadata.csv")

    def test_the_whole_validator_turns_every_failure_into_a_report_error(self):
        for label, raw in (("utf8", b"\xff\xfe\xfa"), ("nul", b"a\x00b\n"),
                           ("quote", b'"unclosed\n'), ("empty", b"")):
            with self.subTest(label=label):
                self.write(raw=raw)
                report = vl.validate_dataset(self.dataset, 1)
                self.assertEqual(len(report["errors"]), 1)
                self.assertEqual(report["stats"], {})

    def test_command_line_exit_code_and_no_traceback(self):
        self.write(raw=b"\xff\xfe\xfa")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = vl.main(["--dataset", str(self.dataset)])
        self.assertEqual(code, 1)
        self.assertIn("ERROR", out.getvalue())
        self.assertIn("RESULT: FAILED", out.getvalue())
        self.assertNotIn("Traceback", out.getvalue() + err.getvalue())
        self.assert_blind(out.getvalue())


# ======================================================= blind and read-only behaviour
class BlindAndReadOnlyTests(MetadataCase):
    BAD_FILES = {
        "duplicate": csv_text([row("secret_P01_001.png"), row("secret_P01_001.png")]),
        "short row": csv_text([["secret_P01_001.png", "P01"]]),
        "bad condition": csv_text([row("secret_P01_001.png", cond="zzz")]),
        "bad sub": csv_text([row("secret_P01_001.png", sub="nope")]),
        "participant": csv_text([row("secret_P01_001.png", pid="NONE")]),
        "blank value": csv_text([row("secret_P01_001.png", when=" ")]),
    }

    def test_no_error_contains_a_filename_a_stem_or_a_path(self):
        for label, text in self.BAD_FILES.items():
            with self.subTest(label=label):
                self.write(text)
                message = self.expect_error("line")
                self.assertNotIn("secret", message)
                report = vl.validate_dataset(self.dataset, 1)
                self.assertNotIn("secret", " ".join(report["errors"]))

    def test_loading_never_writes_anything(self):
        before = {}
        for label, text in self.BAD_FILES.items():
            with self.subTest(label=label):
                self.write(text)
                path = self.dataset / "metadata.csv"
                before[label] = hashlib.sha256(path.read_bytes()).hexdigest()
                try:
                    self.load()
                except ValidationError:
                    pass
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before[label])
                self.assertEqual(sorted(p.name for p in self.dataset.iterdir()), ["metadata.csv"])

    def test_valid_metadata_is_not_modified_either(self):
        self.write(raw=b"\xef\xbb\xbf" + csv_text([row()], eol="\r\n").encode())
        path = self.dataset / "metadata.csv"
        before = path.read_bytes()
        self.load()
        self.assertEqual(path.read_bytes(), before)


# ============================================================= constants stay in sync
class SynchronisationTests(unittest.TestCase):
    def test_conditions_match_the_capture_script(self):
        import capture_dataset as cd
        self.assertEqual(list(vl.CONDITION_SUBS), list(cd.CONDITIONS))
        for cond, info in cd.CONDITIONS.items():
            self.assertEqual(set(vl.CONDITION_SUBS[cond]), set(info["subs"]), cond)
        self.assertEqual(set(vl.PARTICIPANTS), set(cd.PARTICIPANTS))
        self.assertEqual(vl.METADATA_FIELDS, cd.METADATA_FIELDS)
        for cond, info in cd.CONDITIONS.items():                          # who may appear where
            self.assertEqual(info["person"], cond != "bg")

    def test_conditions_match_the_frozen_dedup_script_read_as_text(self):
        source = (SCRIPTS / "build_dedup_manifest.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        order = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "CONDITION_ORDER"
                             for t in node.targets))
        self.assertEqual(order, list(vl.CONDITION_SUBS))

    def test_this_file_never_imports_or_runs_the_dedup_script(self):
        """Order-independent: other test files may import it, so look at THIS file's imports."""
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse([name for name in imported if "build_dedup_manifest" in name])


if __name__ == "__main__":
    unittest.main()
