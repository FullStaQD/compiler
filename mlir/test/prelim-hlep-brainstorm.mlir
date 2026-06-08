
// Really trying to use classical values

func.func @main() {

    %w : i1
    %q0 : !prelimhlep.lin<i1>
    %q1 : !prelimhlep.lin<i1>

    %q0_0, %q1_0 = prelimhlep.delin[%q0 as %c0 : i1] (%q1, %w) {

        scf.if %w {

        }

        prelimhlep.relin()
    }
}
