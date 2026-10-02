// 仕様: 標準入力で受け取った数式の配列を KaTeX で描画し、数式ごとの描画可否を標準出力に返す。
//
// 入力: [{"tex": "\\frac{1}{2}", "display": false}, ...] の JSON
// 出力: 入力と同じ順序・件数の JSON 配列。描画できれば null、できなければ KaTeX のエラーメッセージ
//
// 注意事項:
//   - Python 側（ouj_notebook_converter.evaluation.katex）から 1 回だけ起動される前提で、全数式をまとめて処理する
//   - 日本語など数式内の Unicode 文字は描画できるため、strict モードの警告は無視する
import katex from "katex";

const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}
const formulas = JSON.parse(Buffer.concat(chunks).toString("utf-8"));

const results = formulas.map(({ tex, display }) => {
  try {
    katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: "ignore" });
    return null;
  } catch (error) {
    if (error instanceof katex.ParseError) {
      return error.message;
    }
    throw error;
  }
});

process.stdout.write(JSON.stringify(results));
