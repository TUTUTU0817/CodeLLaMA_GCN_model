
import io.shiftleft.semanticcpg.language.*
import io.shiftleft.codepropertygraph.generated.nodes.*
import java.io.PrintWriter
import ujson.*

// 要先從命令行參數獲取輸出路徑
// val outputPath = "/path/to/your/output.pt"
// 檢查是否有預設的 outputPath 變量，沒有則使用默認值
val finalOutputPath = try {
  outputPath  // 從外部設置的變量
} catch {
  case _: Exception => "bin_graph.json"  // 默認值
}

println(s"📝 輸出路徑: ${finalOutputPath}")

// 節點收集器
val methods = cpg.method.where(_.isExternal(false)).l
println(s"📊 找到 ${methods.size} 個方法")

// 收集所有節點（整個 bin 的所有方法）
val nodes = methods.flatMap { m =>
  val ast = m.ast.l
  val cfg = m.cfgNode.l
  val ddg = cfg.flatMap(_.ddgIn.l)
  val cdg = cfg.flatMap(_._cdgOut.l)
  ast ++ cfg ++ ddg ++ cdg
}.distinct

val nodeToId = nodes.zipWithIndex.toMap
println(s"🔢 總共收集到 ${nodes.size} 個唯一節點")

// 建立節點 JSON
val nodeJson = nodes.map { n =>
  val id = nodeToId(n)
  val code = try {
    n.property("CODE").toString
  } catch {
    case _: Exception => "<no-code>"
  }
  val tpe = n.label
  val fullName = try {
    n.property("FULL_NAME").toString
  } catch {
    case _: Exception => "<unknown>"
  }
  val line = try {
    n.property("LINE_NUMBER").toString
  } catch {
    case _: Exception => ""
  }
  Obj(
    "id" -> Num(id),
    "code" -> Str(code),
    "type" -> Str(tpe),
    "fullName" -> Str(fullName),
    "lineNumber" -> Str(line)
  )
}

// 收集邊（AST、CFG、DDG、CDG）
def makeEdges(label: String, pairs: List[(StoredNode, StoredNode)]) = {
  pairs.collect {
    case (src, dst) if nodeToId.contains(src) && nodeToId.contains(dst) =>
      Obj(
        "source" -> Num(nodeToId(src)),
        "target" -> Num(nodeToId(dst)),
        "type" -> Str(label)
      )
  }
}

// AST
val astEdges = methods.flatMap { m =>
  m.ast.l.flatMap(src => src.astChildren.l.map(dst => (src, dst)))
}

// CFG
val cfgEdges = methods.flatMap { m =>
  m.cfgNode.l.flatMap(src => src.cfgNext.l.map(dst => (src, dst)))
}

// DDG
val ddgEdges = methods.flatMap { m =>
  m.cfgNode.l.flatMap(src => src.ddgIn.l.map(dst => (dst, src)))
}

// CDG
val cdgEdges = methods.flatMap { m =>
  m.cfgNode.l.flatMap(src => src._cdgOut.l.map(dst => (src, dst)))
}

val edgeJson = (makeEdges("AST", astEdges) ++
  makeEdges("CFG", cfgEdges) ++
  makeEdges("DDG", ddgEdges) ++
  makeEdges("CDG", cdgEdges)).distinct

println(s"🔗 總共收集到 ${edgeJson.size} 條邊")

// // 獲取 bin 的基本信息
// val binInfo = try {
// //   val files = cpg.file.name.l
// //   val mainFile = if (files.nonEmpty) files.head else "unknown"
//   Obj(
//     // "binName" -> Str(mainFile),
//     "methodCount" -> Num(methods.size),
//     // "fileCount" -> Num(files.size),
//     "analysisTime" -> Str(java.time.LocalDateTime.now().toString)
//   )
// } catch {
//   case _: Exception => Obj(
//     "binName" -> Str("unknown"),
//     "methodCount" -> Num(methods.size),
//     // "fileCount" -> Num(0),
//     "analysisTime" -> Str(java.time.LocalDateTime.now().toString)
//   )
// }

// // 創建節點類型統計 - 明確轉換為 ujson.Value
// val nodeTypeStatsSeq = nodes.groupBy(_.label).map { case (tpe, nodeList) =>
//   tpe -> Num(nodeList.size)  // 明確轉換為 ujson.Num
// }.toSeq

// // 創建邊類型統計 - 明確轉換為 ujson.Value
// val edgeTypeStatsSeq = edgeJson.groupBy(_.obj("type").str).map { case (tpe, edgeList) =>
//   tpe -> Num(edgeList.size)  // 明確轉換為 ujson.Num
// }.toSeq

// 創建最終的 JSON 結構（整個 bin 為一個單位）
val finalJson = Obj(
//   "binInfo" -> binInfo,
  "nodes" -> Arr(nodeJson),
  "edges" -> Arr(edgeJson),
  "statistics" -> Obj(
    "totalNodes" -> Num(nodeJson.size),
    "totalEdges" -> Num(edgeJson.size),
    "methodCount" -> Num(methods.size)
    // "nodeTypes" -> Obj(nodeTypeStatsSeq*), 
    // "edgeTypes" -> Obj(edgeTypeStatsSeq*) 
  )
)




// 輸出到文件
val writer = new PrintWriter(finalOutputPath)
writer.println(ujson.write(finalJson, indent = 2))
writer.close()

println(s"✅ 已導出 bin 圖數據到 ${finalOutputPath}")
println(s"📊 統計:")
println(s"  - 節點總數: ${nodeJson.size}")
println(s"  - 邊總數: ${edgeJson.size}")
println(s"  - 方法總數: ${methods.size}")

println(s"✅ Exported ${nodeJson.size} nodes and ${edgeJson.size} edges.")
