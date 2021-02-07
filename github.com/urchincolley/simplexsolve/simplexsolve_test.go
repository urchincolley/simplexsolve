package main

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

type TestProblem struct {
	Constraints   []Constraint
	Objective     Objective
	ExpTableau    *Tableau
	ExpTableauErr error
	ExpSolved     *Tableau
	ExpSoln       []float64
	ExpObjVal     float64
}

var cases = map[string]TestProblem{
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
		Objective:  []float64{-2.0, -3.0},
		ExpTableau: &Tableau{mat.NewDense(4, 6, []float64{2, 1, 1, 0, 0, 18, 6, 5, 0, 1, 0, 60, 2, 5, 0, 0, 1, 40, -2, -3, 0, 0, 0, 0})},
		ExpSolved:  &Tableau{mat.NewDense(4, 6, []float64{0, 0, 1, -0.4, 0.2, 2, 1, 0, 0, 0.25, -0.25, 5, 0, 1, 0, -0.1, 0.3, 6, 0, 0, 0, 0.2, 0.4, 28})},
		ExpSoln:    []float64{5.0, 6.0},
		ExpObjVal:  28.0,
	},
	"unequal constraints": {
		Constraints: []Constraint{
			Constraint{
				Coefficients:  []float64{2.0},
				RightHandSide: 18.0,
			}},
		ExpTableauErr: errors.New("coefficient vectors must be of equal length"),
	}}

func roundFloat(x float64) float64 {
	return math.Round(x*100) / 100
}

func TestNewTableau(t *testing.T) {
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			tableau, err := NewTableau(tc.Constraints, tc.Objective)
			if !reflect.DeepEqual(err, tc.ExpTableauErr) {
				t.Errorf("NewTableau error = %e; want %e", err, tc.ExpTableauErr)
				return
			}
			if !reflect.DeepEqual(tableau, tc.ExpTableau) {
				t.Errorf("NewTableau result = %v; want %v", tableau, tc.ExpTableau)
			}
		})
	}
}

func TestSolve(t *testing.T) {
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			sut := tc.ExpTableau
			if sut == nil {
				return
			}
			sut.Solve()
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
						t.Errorf("Solved tableau entry %d,%d = %v; want %v", i, j, entry, expEntry)
					}
				}
			}
		})
	}
}

func TestReadSoln(t *testing.T) {
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			sut := tc.ExpSolved
			if sut == nil {
				return
			}
			soln, val := sut.ReadSoln()
			if !reflect.DeepEqual(soln, tc.ExpSoln) {
				t.Errorf("Solution read = %v; want %v", soln, tc.ExpSoln)
			}
			if val != tc.ExpObjVal {
				t.Errorf("Objective value read = %v; want %v", val, tc.Objective)
			}
		})
	}
}
