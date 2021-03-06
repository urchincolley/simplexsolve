package simplexsolve

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func roundFloat(x float64) float64 {
	return math.Round(x*100) / 100
}

func TestNewTableau(t *testing.T) {
	cases := map[string]struct {
		Constraints   []Constraint
		Objective     Objective
		ExpTableau    *Tableau
		ExpTableauErr error
	}{
		"basic lp": {
			Constraints: []Constraint{
				Constraint{
					Coefficients:  []float64{2.0, 1.0},
					RightHandSide: 18.0,
				},
				Constraint{
					Coefficients:  []float64{6.0, 5.0},
					RightHandSide: 60.0,
				},
				Constraint{
					Coefficients:  []float64{2.0, 5.0},
					RightHandSide: 40.0,
				}},
			Objective: []float64{-2.0, -3.0},
			ExpTableau: &Tableau{mat.NewDense(4, 6, []float64{
				2, 1, 1, 0, 0, 18,
				6, 5, 0, 1, 0, 60,
				2, 5, 0, 0, 1, 40,
				-2, -3, 0, 0, 0, 0})},
		},
		"unequal constraints": {
			Constraints: []Constraint{
				Constraint{
					Coefficients:  []float64{2.0},
					RightHandSide: 18.0,
				}},
			ExpTableauErr: errors.New("coefficient vectors must be of equal length"),
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			tableau, err := NewTableau(tc.Constraints, tc.Objective)
			if !reflect.DeepEqual(err, tc.ExpTableauErr) {
				t.Errorf("NewTableau error = %e; want %e", err, tc.ExpTableauErr)
				return
			}
			if !reflect.DeepEqual(tableau, tc.ExpTableau) {
				t.Errorf("NewTableau result = \n%v;\nwant\n%v",
					mat.Formatted(tableau, mat.Squeeze()),
					mat.Formatted(tc.ExpTableau, mat.Squeeze()))
			}
		})
	}
}

func TestSolve(t *testing.T) {
	cases := map[string]struct {
		UnsolvedTableau *Tableau
		ExpSolved       *Tableau
		ExpSolveErr     error
	}{
		"basic lp": {
			UnsolvedTableau: &Tableau{mat.NewDense(4, 6, []float64{
				2, 1, 1, 0, 0, 18,
				6, 5, 0, 1, 0, 60,
				2, 5, 0, 0, 1, 40,
				-2, -3, 0, 0, 0, 0})},
			ExpSolved: &Tableau{mat.NewDense(4, 6, []float64{
				0, 0, 1, -0.4, 0.2, 2,
				1, 0, 0, 0.25, -0.25, 5,
				0, 1, 0, -0.1, 0.3, 6,
				0, 0, 0, 0.2, 0.4, 28})},
		},
		"unbounded": {
			UnsolvedTableau: &Tableau{mat.NewDense(3, 6, []float64{
				4, -2, 2, 1, 0, 4,
				2, -1, 1, 0, 1, 1,
				-3, -2, 5, 0, 0, 0})},
			ExpSolved: &Tableau{mat.NewDense(3, 6, []float64{
				0, 0, 0, 1, -2, 2,
				1, -0.5, 0.5, 0, 0.5, 0.5,
				0, -3.5, 6.5, 0, 1.5, 1.5})},
			ExpSolveErr: ERR_UNBOUNDED,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			sut := tc.UnsolvedTableau
			err := sut.Solve()
			if !reflect.DeepEqual(err, tc.ExpSolveErr) {
				t.Errorf("Solve() error = %v; want %v", err, tc.ExpSolveErr)
			}

			rs, cs := sut.Dims()
			ers, ecs := tc.ExpSolved.Dims()
			if rs != ers || cs != ecs {
				t.Errorf("Solved tableau dims = %d, %d; want %d, %d", rs, cs, ers, ecs)
				return
			}

			for i := 0; i < rs; i++ {
				for j := 0; j < cs; j++ {
					entry := roundFloat(sut.At(i, j))
					expEntry := tc.ExpSolved.At(i, j)
					if entry != expEntry {
						t.Errorf("Solved tableau = \n%v;\nwant\n%v",
							mat.Formatted(sut, mat.Squeeze()),
							mat.Formatted(tc.ExpSolved, mat.Squeeze()))
						return
					}
				}
			}
		})
	}
}

func TestReadSoln(t *testing.T) {
	cases := map[string]struct {
		UnsolvedTableau *Tableau
		SolvedTableau   *Tableau
		ExpSoln         []float64
		ExpObjVal       float64
		ExpReadSolnErr  error
	}{

		"basic lp": {
			UnsolvedTableau: &Tableau{mat.NewDense(4, 6, []float64{
				2, 1, 1, 0, 0, 18,
				6, 5, 0, 1, 0, 60,
				2, 5, 0, 0, 1, 40,
				-2, -3, 0, 0, 0, 0})},
			SolvedTableau: &Tableau{mat.NewDense(4, 6, []float64{
				0, 0, 1, -0.4, 0.2, 2,
				1, 0, 0, 0.25, -0.25, 5,
				0, 1, 0, -0.1, 0.3, 6,
				0, 0, 0, 0.2, 0.4, 28})},
			ExpSoln:   []float64{5.0, 6.0},
			ExpObjVal: 28.0,
		},
		"unbounded": {
			UnsolvedTableau: &Tableau{mat.NewDense(3, 6, []float64{
				4, -2, 2, 1, 0, 4,
				2, -1, 1, 0, 1, 1,
				-3, -2, 5, 0, 0, 0})},
			SolvedTableau: &Tableau{mat.NewDense(3, 6, []float64{
				0, 0, 0, 1, -2, 2,
				1, -0.5, 0.5, 0, 0.5, 0.5,
				0, -3.5, 6.5, 0, 1.5, 1.5})},
			ExpReadSolnErr: ERR_UNBOUNDED,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			sut1 := tc.UnsolvedTableau
			_, _, err := sut1.ReadSoln()
			if !reflect.DeepEqual(err, ERR_UNSOLVED) {
				t.Errorf("ReadSoln() for unsolved tableau = %v; want %v", err, ERR_UNSOLVED)
			}

			sut2 := tc.SolvedTableau
			soln, val, err := sut2.ReadSoln()
			if !reflect.DeepEqual(err, tc.ExpReadSolnErr) {
				t.Errorf("ReadSoln() for solved tableau = %v; want %v", err, tc.ExpReadSolnErr)
			}
			if !reflect.DeepEqual(soln, tc.ExpSoln) {
				t.Errorf("Solution read = %v; want %v", soln, tc.ExpSoln)
			}
			if val != tc.ExpObjVal {
				t.Errorf("Objective value read = %v; want %v", val, tc.ExpObjVal)
			}
		})
	}
}
