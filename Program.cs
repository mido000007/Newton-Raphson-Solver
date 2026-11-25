#nullable disable
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Linq;
using System.Windows.Forms;

// ============================================================
//  Newton-Raphson Solver  (Windows Forms - .NET 6 or later)
//  x(i+1) = x(i) - f(x(i)) / f'(x(i))
// ============================================================

static class Program
{
    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.Run(new MainForm());
    }
}

// ---------- 1) Expression parser: turns text like "x^3 - 8x + 5" into a function ----------
class Parser
{
    readonly string s;
    int p;
    Parser(string text) { s = text; }

    public static Func<double, double> Compile(string text)
    {
        var ps = new Parser(text);
        var e = ps.Expr();
        ps.Ws();
        if (ps.p < ps.s.Length)
            throw new FormatException($"Unexpected symbol '{ps.s[ps.p]}' at position {ps.p + 1}");
        return e;
    }

    void Ws() { while (p < s.Length && char.IsWhiteSpace(s[p])) p++; }

    bool Eat(char c)
    {
        Ws();
        if (p < s.Length && s[p] == c) { p++; return true; }
        return false;
    }

    // expr = term { (+|-) term }
    Func<double, double> Expr()
    {
        var l = Term();
        while (true)
        {
            if (Eat('+')) { var a = l; var r = Term(); l = x => a(x) + r(x); }
            else if (Eat('-')) { var a = l; var r = Term(); l = x => a(x) - r(x); }
            else return l;
        }
    }

    // term = unary { (*|/|implicit) unary }
    Func<double, double> Term()
    {
        var l = Unary();
        while (true)
        {
            if (Eat('*')) { var a = l; var r = Unary(); l = x => a(x) * r(x); }
            else if (Eat('/')) { var a = l; var r = Unary(); l = x => a(x) / r(x); }
            else
            {
                Ws();
                // implicit multiplication: 8x  or  2(x+1)
                if (p < s.Length && (char.IsLetter(s[p]) || s[p] == '('))
                { var a = l; var r = Unary(); l = x => a(x) * r(x); }
                else return l;
            }
        }
    }

    Func<double, double> Unary()
    {
        if (Eat('-')) { var u = Unary(); return x => -u(x); }
        if (Eat('+')) return Unary();
        return Power();
    }

    // power = primary [ ^ unary ]   (right associative)
    Func<double, double> Power()
    {
        var b = Primary();
        if (Eat('^')) { var e = Unary(); return x => Math.Pow(b(x), e(x)); }
        return b;
    }

    Func<double, double> Primary()
    {
        Ws();
        if (p >= s.Length) throw new FormatException("The expression ended unexpectedly");
        char c = s[p];

        if (char.IsDigit(c) || c == '.')
        {
            int st = p;
            while (p < s.Length && (char.IsDigit(s[p]) || s[p] == '.')) p++;
            double v = double.Parse(s.Substring(st, p - st), CultureInfo.InvariantCulture);
            return x => v;
        }

        if (c == '(')
        {
            p++;
            var e = Expr();
            if (!Eat(')')) throw new FormatException("Missing closing parenthesis )");
            return e;
        }

        if (char.IsLetter(c))
        {
            int st = p;
            while (p < s.Length && char.IsLetter(s[p])) p++;
            string name = s.Substring(st, p - st).ToLowerInvariant();

            if (name == "x") return x => x;
            if (name == "pi") return x => Math.PI;
            if (name == "e") return x => Math.E;

            Func<double, double> fn = name switch
            {
                "sin" => Math.Sin,
                "cos" => Math.Cos,
                "tan" => Math.Tan,
                "asin" => Math.Asin,
                "acos" => Math.Acos,
                "atan" => Math.Atan,
                "exp" => Math.Exp,
                "ln" => Math.Log,
                "log" => Math.Log10,
                "sqrt" => Math.Sqrt,
                "abs" => Math.Abs,
                _ => throw new FormatException($"Unknown name '{name}'")
            };
            if (!Eat('(')) throw new FormatException($"Function '{name}' needs parentheses");
            var arg = Expr();
            if (!Eat(')')) throw new FormatException("Missing closing parenthesis )");
            return x => fn(arg(x));
        }

        throw new FormatException($"Unexpected symbol '{c}' at position {p + 1}");
    }
}

// ---------- 2) The Newton-Raphson engine ----------
class NewtonResult
{
    public List<double> X = new List<double>();       // x0, x1, x2, ...
    public List<double> Fx = new List<double>();      // f(x_i)
    public List<double> Err = new List<double>();     // |x(i+1) - x(i)|
    public bool Converged;
    public int Iterations;
    public double ConvergenceTest;                    // |f f'' / f'^2| at x0
}

static class Newton
{
    public static double Derivative(Func<double, double> f, double x)
    {
        double h = 1e-6 * Math.Max(1, Math.Abs(x));
        return (f(x + h) - f(x - h)) / (2 * h);
    }

    public static double SecondDerivative(Func<double, double> f, double x)
    {
        double h = 1e-4 * Math.Max(1, Math.Abs(x));
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
    }

    public static NewtonResult Solve(Func<double, double> f, Func<double, double> df,
                                     double x0, double tol, int maxIter = 100)
    {
        var r = new NewtonResult();
        double x = x0;
        r.X.Add(x); r.Fx.Add(f(x)); r.Err.Add(double.NaN);

        double d0 = df(x);
        r.ConvergenceTest = Math.Abs(f(x) * SecondDerivative(f, x) / (d0 * d0));

        for (int i = 1; i <= maxIter; i++)
        {
            double d = df(x);
            if (Math.Abs(d) < 1e-12)
                throw new InvalidOperationException($"f'(x) = 0 at x = {x:G6}. Try a different x0.");

            double xn = x - f(x) / d;
            if (double.IsNaN(xn) || double.IsInfinity(xn))
                throw new InvalidOperationException("The values diverged. Try a different x0.");

            double err = Math.Abs(xn - x);
            r.X.Add(xn); r.Fx.Add(f(xn)); r.Err.Add(err);
            r.Iterations = i;
            x = xn;

            if (err < tol) { r.Converged = true; break; }
        }
        return r;
    }
}

// ---------- 3) The window ----------
class MainForm : Form
{
    TextBox txtF, txtDf, txtX0, txtTol;
    Button btnSolve;
    DataGridView grid;
    Label lblCond, lblResult;
    Panel plot;

    Func<double, double> f;
    NewtonResult res;

    public MainForm()
    {
        Text = "Newton-Raphson Solver";
        Width = 1150; Height = 700;
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 10);

        // --- top panel (inputs) ---
        var top = new Panel { Dock = DockStyle.Top, Height = 135 };
        Controls.Add(top);

        AddLabel(top, "f(x) =", 10, 15);
        txtF = new TextBox { Left = 70, Top = 12, Width = 330, Text = "x^3 - 8x + 5" };
        top.Controls.Add(txtF);

        AddLabel(top, "f'(x) (optional) =", 420, 15);
        txtDf = new TextBox { Left = 570, Top = 12, Width = 280 };
        top.Controls.Add(txtDf);

        AddLabel(top, "x0 =", 10, 52);
        txtX0 = new TextBox { Left = 70, Top = 49, Width = 80, Text = "0" };
        top.Controls.Add(txtX0);

        AddLabel(top, "Tolerance =", 170, 52);
        txtTol = new TextBox { Left = 260, Top = 49, Width = 90, Text = "0.000001" };
        top.Controls.Add(txtTol);

        btnSolve = new Button { Left = 370, Top = 46, Width = 120, Height = 32, Text = "Solve" };
        btnSolve.Click += (s, e) => Solve();
        top.Controls.Add(btnSolve);
        AcceptButton = btnSolve;

        lblCond = new Label { Left = 10, Top = 88, AutoSize = true };
        lblResult = new Label { Left = 10, Top = 110, AutoSize = true, Font = new Font("Segoe UI", 10, FontStyle.Bold) };
        top.Controls.Add(lblCond);
        top.Controls.Add(lblResult);

        // --- table (left) and plot (right) ---
        grid = new DataGridView
        {
            Dock = DockStyle.Left,
            Width = 460,
            ReadOnly = true,
            AllowUserToAddRows = false,
            RowHeadersVisible = false,
            AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
            BackgroundColor = Color.White
        };
        grid.Columns.Add("i", "i");
        grid.Columns.Add("x", "x(i)");
        grid.Columns.Add("fx", "f(x(i))");
        grid.Columns.Add("err", "|x(i+1) - x(i)|");
        grid.Columns[0].FillWeight = 25;

        plot = new DoubleBufferedPanel { Dock = DockStyle.Fill, BackColor = Color.White };
        plot.Paint += (s, e) => DrawPlot(e.Graphics, plot.ClientRectangle);
        plot.Resize += (s, e) => plot.Invalidate();

        Controls.Add(plot);
        Controls.Add(grid);
        top.BringToFront(); // keep the top panel above
        Controls.SetChildIndex(top, Controls.Count - 1);
    }

    static void AddLabel(Control parent, string text, int x, int y)
        => parent.Controls.Add(new Label { Text = text, Left = x, Top = y, AutoSize = true });

    static double ParseNumber(string t)
        => double.Parse(t.Trim().Replace(',', '.'), CultureInfo.InvariantCulture);

    void Solve()
    {
        try
        {
            f = Parser.Compile(txtF.Text);
            Func<double, double> df = string.IsNullOrWhiteSpace(txtDf.Text)
                ? (x => Newton.Derivative(f, x))
                : Parser.Compile(txtDf.Text);

            double x0 = ParseNumber(txtX0.Text);
            double tol = ParseNumber(txtTol.Text);
            if (tol <= 0) throw new FormatException("Tolerance must be greater than 0");

            res = Newton.Solve(f, df, x0, tol);

            // fill the table
            grid.Rows.Clear();
            for (int i = 0; i < res.X.Count; i++)
            {
                grid.Rows.Add(i,
                    res.X[i].ToString("F8", CultureInfo.InvariantCulture),
                    res.Fx[i].ToString("E3", CultureInfo.InvariantCulture),
                    i == 0 ? "----" : res.Err[i].ToString("E3", CultureInfo.InvariantCulture));
            }

            lblCond.Text = $"Convergence condition at x0: |f·f'' / f'^2| = {res.ConvergenceTest:F5}  " +
                           (res.ConvergenceTest < 1 ? "(< 1  ✔ converges)" : "(≥ 1  ✘ may not converge)");

            double root = res.X[res.X.Count - 1];
            lblResult.Text = res.Converged
                ? $"Number of iterations = {res.Iterations}     Root ≈ {root:F6}     f(root) = {res.Fx[res.Fx.Count - 1]:E3}"
                : $"Did not converge after {res.Iterations} iterations (last x = {root:F6})";
            lblResult.ForeColor = res.Converged ? Color.DarkGreen : Color.Firebrick;

            plot.Invalidate();
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
    }

    // ---------- plotting with GDI+ ----------
    void DrawPlot(Graphics g, Rectangle rc)
    {
        g.Clear(Color.White);
        if (res == null || f == null) return;
        g.SmoothingMode = SmoothingMode.AntiAlias;

        const int M = 45;
        var area = new Rectangle(M, M / 2, rc.Width - M - 15, rc.Height - M - M / 2);
        if (area.Width < 50 || area.Height < 50) return;

        // x range around the iterates
        double lo = res.X.Min(), hi = res.X.Max();
        double pad = Math.Max((hi - lo) * 0.3, 1.0);
        lo -= pad; hi += pad;

        // y range
        double m = res.X.Select(v => Math.Abs(f(v))).Where(v => !double.IsNaN(v) && !double.IsInfinity(v))
                        .DefaultIfEmpty(1).Max() * 1.3;
        if (m < 1e-9) m = 1;

        double cmin = double.MaxValue, cmax = double.MinValue;
        const int N = 800;
        var ys = new double[N + 1];
        for (int i = 0; i <= N; i++)
        {
            double y = f(lo + (hi - lo) * i / N);
            ys[i] = y;
            if (!double.IsNaN(y) && !double.IsInfinity(y)) { cmin = Math.Min(cmin, y); cmax = Math.Max(cmax, y); }
        }
        double ymin = Math.Max(Math.Min(cmin, 0), -3 * m);
        double ymax = Math.Min(Math.Max(cmax, 0), 3 * m);
        double yr = (ymax - ymin) * 0.1; ymin -= yr; ymax += yr;

        float SX(double x) => (float)(area.Left + (x - lo) / (hi - lo) * area.Width);
        float SY(double y) => (float)(area.Bottom - (y - ymin) / (ymax - ymin) * area.Height);

        // frame + axes
        g.DrawRectangle(Pens.LightGray, area);
        using (var axis = new Pen(Color.Black, 1.5f))
        {
            g.DrawLine(axis, area.Left, SY(0), area.Right, SY(0));
            if (lo < 0 && hi > 0) g.DrawLine(axis, SX(0), area.Top, SX(0), area.Bottom);
        }
        using (var br = new SolidBrush(Color.DimGray))
        {
            g.DrawString(lo.ToString("G4"), Font, br, area.Left - 10, area.Bottom + 5);
            g.DrawString(hi.ToString("G4"), Font, br, area.Right - 30, area.Bottom + 5);
            g.DrawString(ymax.ToString("G3"), Font, br, 2, area.Top);
            g.DrawString(ymin.ToString("G3"), Font, br, 2, area.Bottom - 18);
        }

        g.SetClip(area);

        // the curve
        using (var pen = new Pen(Color.RoyalBlue, 2f))
        {
            for (int i = 0; i < N; i++)
            {
                if (double.IsNaN(ys[i]) || double.IsInfinity(ys[i]) ||
                    double.IsNaN(ys[i + 1]) || double.IsInfinity(ys[i + 1])) continue;
                float y1 = SY(ys[i]), y2 = SY(ys[i + 1]);
                if (Math.Abs(y1) > 1e5 || Math.Abs(y2) > 1e5) continue;
                g.DrawLine(pen, SX(lo + (hi - lo) * i / N), y1, SX(lo + (hi - lo) * (i + 1) / N), y2);
            }
        }

        // tangent steps: (x_i, f(x_i)) -> (x_{i+1}, 0)
        using (var dash = new Pen(Color.Gray, 1f) { DashStyle = DashStyle.Dash })
        {
            for (int i = 0; i < res.X.Count - 1; i++)
            {
                g.DrawLine(dash, SX(res.X[i]), SY(0), SX(res.X[i]), SY(res.Fx[i]));
                g.DrawLine(dash, SX(res.X[i]), SY(res.Fx[i]), SX(res.X[i + 1]), SY(0));
            }
        }

        // the points
        using (var red = new SolidBrush(Color.Crimson))
        using (var txt = new SolidBrush(Color.Black))
        {
            for (int i = 0; i < res.X.Count; i++)
            {
                float px = SX(res.X[i]), py = SY(res.Fx[i]);
                g.FillEllipse(red, px - 4, py - 4, 8, 8);
                g.DrawString("x" + i, Font, txt, px + 5, py - 18);
            }
            // the root on the x-axis
            using (var green = new SolidBrush(Color.ForestGreen))
            {
                float rx = SX(res.X[res.X.Count - 1]), ry = SY(0);
                g.FillEllipse(green, rx - 6, ry - 6, 12, 12);
            }
        }
        g.ResetClip();
    }
}

class DoubleBufferedPanel : Panel
{
    public DoubleBufferedPanel() { DoubleBuffered = true; ResizeRedraw = true; }
}