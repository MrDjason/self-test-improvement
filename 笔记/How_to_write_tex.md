

# LaTeX 排版

最重要的一点：哪不懂，问ai

## 入门

### 文档结构

```latex
\documentclass{artical}
%导言区
\begin{document}
%正文
\end{document}
```

### 导言区

导入宏包(可一次调用多个)

```latex
\usepachage{tabularx, makecell, multirow}
```

文件的组织方式

```latex
include命令（新开一页）
\include{〈filename〉}
\include{chapters/file}%相对路径
\include{/home/Bob/file}%*nix(包含Linux、macOS)绝对路径
\include{D:/file}%Windows绝对路径,用正斜线
```

```latex
input命令（不另起一页）
```

syntonly宏包（只排查错误，不生成PDF）

```latex
\usepachage{syntpnly}
\syntaxonly
```

## 文字、章节、引用

### 空格与分段

多个空格视为一个空格，行末的换行符视为空格，空行视为分段，也可以使用\par分段 。

```latex
Just note that the content is made up and might not have a specific context in a real-world sense. \par The gentle breeze rustled through the tall grass in the meadow, carrying with it the faint scent of wildflowers. A small, colorful butterfly flitted from one blossom to another, its wings shimmering in the warm sunlight. 
或
Just note that the content is made up and might not have a specific context in a real-world sense. 

The gentle breeze rustled through the tall grass in the meadow, carrying with it the faint scent of wildflowers. A small, colorful butterfly flitted from one blossom to another, its wings shimmering in the warm sunlight. 
```

verb命令：将两条线之间的部分原样输出

```latex
ust note that the content is made up and might not have a specific context in a real-world sense. \verb|\par| The gentle breeze rustled through the tall grass in the meadow, carrying with it the faint scent of wildflowers.
输出：
ust note that the content is made up and might not have a specific context in a real-world sense. \par The gentle breeze rustled through the tall grass in the meadow, carrying with it the faint scent of wildflowers.
```

注释

```latex
%注释
```

转义字符

```latex
50\% 输出 50%
\textbackslash：输出\(不能用\\ ，因为\\是换行符)
```

断行（不会分段，没有缩进）

```latex
\\[<length>]可选参数
\\*[<length>]禁止在断行处分页
\newline不带参数
```

断页

```latex
\newpage（对于两栏的文章，可以分栏）
\clearpage
```

断词

```latex
（一般自动生成）
```

### 标题

一篇结构化的、条理清晰文档一定是层次分明的，通过不同的命令分割为章、节、小节。三个标准文档类article、report和book¹提供了划分章节的命令

```latex
\chapter{〈title〉} 
\section{〈title〉}一级标题
\subsection{〈title〉}二级标题
\subsubsection{(title〉}
\paragraph{(title)}
\subparagraph{〈title〉}
其中\chapter只在report和book文档类有定义。这些命令生成章节标题，并能够自动编号。
```

除此之外ATEX还提供了\part命令，用来将整个文档分割为大的分块，但不影响\chapter或\section等的编号。

上述命令除了生成带编号的标题之外，还向目录中添加条目，并影响页眉页脚的内容。每个命令有两种变体：

```latex
●带可选参数的变体:\section[〈shorttitle〉]{〈title〉}
标题使用(title)参数,在目录和页眉页脚中使用(shorttitle)参数;
●带星号的变体:\section*{〈title〉}
标题不带编号（比如2.1），也不生成目录项和页眉页脚。
```

较低层次如`\paragraph`和`\subparagraph`即使不用带星号的变体，生成的标题默认也不带编号,事实上,除`\part`外:

```latex
●article文档类带编号的层级为\section、\subsection、\subsubsection三级;
●report和book文档类带编号的层级为\chapter、\section、\subsection三级。
```

### 目录

生成目录非常容易，只需在合适的地方使用命令`\tableofcontents`

```latex
这个命令会生成单独的一章(report/book)或一节(article),标题默认为“Contents”,可通过8.4节给出的方法定制标题。\tableofcontents生成的章节默认不写入目录(\section*或\chapter*),可使用tocbibind等宏包修改设置。正确生成目录项，一般需要编译两次源代码。
有时我们使用了\chapter*或\section*这样不生成目录项的章节标题命令，而又想手动生成该章节的目录项，可以在标题命令后面使用：\addcontentsline{toc}{〈level〉}{〈title〉}其中〈level〉为章节层次chapter或section等,〈title〉为出现于目录项的章节标题。titletoc、tocloft等宏包提供了具体定制目录项格式的功能，详情请参考宏包的帮助文档。
```

### 附录

所有标准文档类都提供了一个`\appendix`命令将正文和附录分开，使用`\appendix`后，最高一级章节编号为按ABCD编号。

```latex
\appendix
	\section{附录 A}
	这是第一个附录的内容。
	
	\section{附录 B}
	这是第二个附录的内容。
```



### 标题页

latex支持生成简单的标题页。首先需要给定标题和作者等信息。

```latex
这部分要在导言区
\title{〈title〉}  
\author{〈author〉} 
\date{〈date〉}
```

```latex
其中前两个命令是必须的(不用\title会报错;不用\author会警告),\date命令可选。
LATEX还提供了一个\today命令自动生成当前日期,\date默认使用\today。在\title、\author等命令内可以使用\thanks命令生成标题页的脚注，用\and隔开多个人名。在信息给定后，就可以使用\maketitle命令生成一个简单的标题页了。
article文档类的标题默认不单独成页，而report和book默认单独成页。可在\documentclass命令调用文档类时指定titlepage或notitlepage选项以修改默认的行为。LATEX标准类还提供了一个简单的titlepage环境，生成不带页眉页脚的一页。用户可以在这个环境中使用各种排版元素自由发挥，生成自定义的标题页以替代\maketitle命令。甚至可以利用titlepage环境重新定义\maketitle
```

```latex
\maketitle须在正文区内
```

### 交叉引用

使用标签`\lable{}`和`\ref{}`（引用章）或`\pageref`（引用页）（插入一个n，n为被应用的章节数或页数）

```latex
The gentle breeze rustled the leaves of the tall oak trees in the park. People were strolling along the winding paths, some with their dogs trotting beside them. Children were laughing and playing on the grassy areas, flying colorful kites that danced in the sky. A small pond nearby was filled with lilies, their soft petals floating on the calm water. \lable{be_labeled}The sun peeked through the clouds, casting a warm glow over everything, creating a serene and picturesque scene.
引用时：（需编译两次）
In the small coastal town, the smell of saltwater mingled with the scent of freshly baked bread from the local baker\ref{be_labeled}. Seagulls soared overhead, their cries echoing against the backdrop of the crashing waves. Fishermen were busy unloading their catch of the day at the dock, chatting animatedly with one another. Along the sandy shore, tourists were building sandcastles, their faces alight with joy.
```

### 脚注与边注

使用`\footnote`命令可以在页面底部生成一个脚注：

```
\footnote{(footnote)}
```

```latex
假如我们输入以下文字和命令：
“天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。”\footnote{出自《千字文》。}
在正文中则为：“天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。”
有些情况下(比如在表格环境、各种盒子内)使用\footnote并不能正确生成脚注。我们可以分两步进行,先使用\footnotemark为脚注计数,再在合适的位置用\footnotetext生成脚注。
使用\marginpar命令可在边栏位置生成边注：\marginpar[〈left-margin〉]{〈right-margin〉}
```

### 列表

列表的形式：

```latex
\begin{enumerate}%（编号前是序号）
	\item 第一
		\begin{enumerate}
			\item 第一（第二级列表默认是abcd编号，可用可选参数修改。例如\item[*]）
		\end{enumerate}
	\item 第二
	\item 第三

\end{enumerate}

```

```latex
\begin{itemize}%（编号前是圆点）
\item 第一
		\begin{enumerate}
			\item 第一（第二级列表默认是123编号）
		\end{enumerate}
	\item 第二
	\item 第三

\end{itemize}
```

```latex
\begin{description}
\item [第一]abc(被方括号括起来的会被加粗，放在序号的位置上)
		\begin{enumerate}
			\item 第一（第二级列表默认是123编号）
		\end{enumerate}
	\item 第二
	\item 第三


\end{description}
```

### 对齐环境

center、 flushleft 和 flushright环境分别用于生成居中，左对齐和右对齐的文本环境。

```latex
\begin{center}...\end{center}
\begin{flushleft)...\end{flushleft}
\begin{flushright}... \end{flushright}
```

```latex
\begin{center}
Centered text using a\verb|center| environment.
\end{center}
\begin{flushleft}
Left-aligned text using a\verb|flushleft| environment.
\end{flushleft}
\begin{flushright}
Right-aligned text using a\verb|flushright| environment.
\end{flushright}
```

除此之外，还可以用以下命令直接改变文字的对齐方式：

这种方式会把后面所有的内容对齐，直到遇到新的对齐命令。

```latex
\centering  \raggedright  \raggedleft
```

```latex
\centering
Centered text paragraph.
\raggedright
Left-aligned text paragraph.
\raggedleft
Right-alignedtext paragraph.
```

### 引用环境

了解即可，用的不多

### 摘要环境

```latex
\begin{abstract}
午后的时光慵懒而惬意，那只毛色斑驳的花猫正蜷卧在洒满阳光的窗台上，偶尔抖动一下耳朵，似是被梦中的小鱼干惊扰。不远处的花园里，不知名的花朵肆意绽放着，红的热烈、粉的娇嫩、黄的明艳，像是一群身着华服的精灵在翩翩起舞。微风轻拂，带着泥土的芬芳和花朵的甜香，悠悠地飘进屋内。此时，桌上的旧唱片机正缓缓转动，发出沙沙的声响，那是岁月留下的独特旋律，仿佛在诉说着那些被时光掩埋的故事。而在世界的某个角落，一场奇妙的冒险或许正悄然拉开帷幕，勇敢的少年带着坚定的目光，踏上未知的旅程，去探寻那隐藏在迷雾背后的神秘宝藏。 
\end{abstract}
```

### 代码环境

```latex
\begin{verbatim}
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}    
\end{verbatim}
```

```latex
\begin{verbatim*}%(将空格显示成_)
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}    
\end{verbatim*}
```

## 表格与图片

### 表格

LaTeX原版表格不太好写，使用其他工具生成

[在线表格网站：http://latex-tables.com](http://latex-tables.com)

三线表需要包：

```latex
\usepackage{booktabs}
其他的注释啥加上啥
```

网站画三线表：划线分别选toprule,midrule,bottomrule

### 图片

需要宏包

```latex
\usepackage{graphicx}
```

通过导航栏-向导-插入图片插入

## 盒子和浮动体

### 浮动体

内容丰富的文章或者书籍往往包含许多图片和表格等内容。这些内容的尺寸往往太大，导致分页困难。LATEX为此引入了浮动体的机制，令大块的内容可以脱离上下文，放置在合适的位置。LATEX预定义了两类浮动体环境figure和table。习惯上figure里放图片, table里放表格，但并没有严格限制，可以在任何一个浮动体里放置文字、公式、表格、图片等等任意内容。

```latex
\begin{table}[〈placement〉]
…
\end{table}
〈placement〉参数提供了一些符号用来表示浮动体允许排版的位置，如hbp允许浮动体排版在当前位置、底部或者单独成页。table和figure浮动体的默认设置为tbp。
```

```latex
\caption的用法非常类似于\section等命令,可以用带星号的命令\caption*生成不带编号的标题，也可以使用带可选参数的形式\caption[…]{…}，使得在目录里使用短标题。\caption命令之后还可以紧跟\label命令标记交叉引用。
\caption生成的标题形如“Figure1:…”(figure环境)或“Table1:…”(table环境)。可通过修改\figurename和\tablename的内容来修改标题的前缀。标题样式的定制功能由caption宏包提供，详见该宏包的帮助文档，在此不作赘述。table和figure两种浮动体分别有各自的生成目录的命令：
\listoftables
\listoffigures
它们类似\tableofcontents生成单独的章节。
```

`float` 宏包提供了一个 `H` 选项，它可以强制将浮动体放在代码指定的位置。首先需要在文档导言区引入 `float` 宏包，然后在 `figure` 环境中使用 `H` 选项。

```latex
\usepackage{float}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.5\textwidth]{example-image}
    \caption{This is an example image.}
\end{figure}
```

### 盒子

生成水平盒子的命令如下：

```latex
\mbox{...}
\makebox[〈width〉][〈align〉]{...}
```

`\mbox`生成一个基本的水平盒子，内容只有一行，不允许分段(除非嵌套其它盒子，比如后文的垂直盒子)。外表看上去，`\mbox`的内容与正常的文本无二，不过断行时文字不会从盒子里断开。`\makebox`更进一步，可以加上可选参数用于控制盒子的宽度`〈width〉`，以及内容的对齐方式`〈align〉`,可选居中c(默认值)、左对齐1、右对齐r和分散对齐s。

```latex
|\mbox{Test some words.}|\\				
|\makebox[10em]{Test  some words.}|\\				
|\makebox[10em][1]{Test  some words.}|\\
|\makebox[10em][r]{Test  some words.}|\\
|\makebox[10em][s]{Test  some words.}|
```

若要打印边框：

```latex
\fbox{…}
\framebox[〈width〉][〈align〉]{...}
```

```latex
\fbox{Test  some words.}\\
\framebox[10em][r]{Test  some  words.}
```

可以通过`\setlength`命令调节边框的宽度`\fboxrule`和内边距`\fboxsep`:

```latex
\framebox[10em][r]{Test  box}\\[lex]
\setlength{\fboxrule}{1.6pt}
\setlength{\fboxsep}{1em}
\framebox[10em][r]{Test  box}
```

### 可排版盒子

如果需要排版一个文字可以换行的盒子，LATEX提供了两种方式：

```latex
\parbox[〈align〉][〈height〉][〈inner-align〉]{〈width〉}{...}
或
\begin{minipage}[〈align〉][〈height〉][〈inner-align〉]{〈width〉}
…
\end{minipage}
```

其中`〈align〉`为盒子和周围文字的对齐情况(类似tabular环境);`〈height〉`和`〈inner-align〉`设置盒子的高度和内容的对齐方式，类似水平盒子`\makebox`的设置，不过`〈inner-align〉`接受的参数是顶部t、底部b、居中c和分散对齐s。

|特性|Parbox命令|Minipage环境|
|---|---|---|
|用途|创建固定宽度的段落盒子，适合单行或简单文本|创建块级结构，适合更复杂的布局|
|内容处理|适合简单内容，内容不会自动换行或布局复杂元素|支持段落、换行、并排元素等复杂布局|
|换行|内容不会自动换行，除非手动指定|内部内容会自动换行和处理段落|
|位置选项|可以设置垂直对齐（vtop、vcenter、vbottom）|可以设置垂直对齐（t、c、b）|
|复杂布局|适合简单的布局，不能轻松实现复杂的元素组合|支持多个元素的并排显示，适合复杂布局|
|自动段落|不会自动生成段落|会自动生成段落和处理段间距|

### 并排和子图表

方法一：前期处理（ppt）

方法二：将两个或多个图打包放到一个盒子里，再对盒子命名。

```latex
并排图
\begin{figure}[htbp]
    \centering
    \includegraphics[width=...(0.45\textwidth 页面宽0.45倍)]{...}%图片1
    \qquad%图片1和图片2并排中间的空格，几个q几个空格
    \includegraphics[width=...]{...}\\[...pt]%图片2
    \includegraphics[width=...]{...}%图片3
    \caption{...}
\end{figure}
子图表：图1(a),图1(b),图1(c)
%需要subcaption包
\begin{figure}[htbp]
    \centering
    \begin{subfigure}{...}
    \centering
    \includegraphics[width=...]{...}
    \caption{...}
    \end{subfigure}
    \qquad
    \begin{subfigure}{...}
    \centering
    \includegraphics[width=...]{...}
    \caption{...}
    \end{subfigure}
\end{figure}
```

豆包：下面是一个通用的 LaTeX 代码模板，它能用于在文档中纵向排列多张图片，并且每张图片带有标题和标签。

```latex
\documentclass{article}
\usepackage{graphicx} % 用于插入图片
\usepackage{subcaption} % 用于创建带有子标题的子图
\usepackage{float} % 若需要固定图片位置，可使用 [H] 选项

\begin{document}

\begin{figure}[H] % [H] 选项让图片固定在代码位置，若不需要可移除
    \centering % 使整个图片组居中
    % 第一个子图
    \begin{subfigure}{\linewidth} % 子图宽度为整行宽度
        \centering % 子图内容居中
        \includegraphics[width=0.8\linewidth]{path/to/your/image1.png} % 插入图片，宽度为子图宽度的 80%
        \caption{第一个子图的标题} % 子图标题
        \label{fig:image1} % 子图标签，方便引用
    \end{subfigure}
    % 第二个子图
    \begin{subfigure}{\linewidth}
        \centering
        \includegraphics[width=0.8\linewidth]{path/to/your/image2.png}
        \caption{第二个子图的标题}
        \label{fig:image2}
    \end{subfigure}
    % 第三个子图（可按需添加更多子图）
    \begin{subfigure}{\linewidth}
        \centering
        \includegraphics[width=0.8\linewidth]{path/to/your/image3.png}
        \caption{第三个子图的标题}
        \label{fig:image3}
    \end{subfigure}
    \caption{整个图片组的主标题} % 整个 figure 的主标题
    \label{fig:overall} % 整个 figure 的标签，方便引用
\end{figure}

\end{document}
```



## 公式与参考文献

### 公式

需要宏包amsmath

行内公式

```latex
$公式$
```

行间公式

单独成行的行间公式由equation环境包裹。equation环境为公式自动生成一个编号,这个编号可以用``\label`和`\ref`生成交叉引用, amsmath的`\eqref`命令甚至为引用自动加上圆括号；还可以用`\tag`命令手动修改公式的编号，或者用`\notag`命令取消为公式编号(与之基本等效的命令是`\nonumber`)。

```latex
\begin{equation}
a^2 + b^2 = c^2\label{pythagorean}(\notage（不编号）)
\end{equation}
```

引用时使用`\eqref{pythagorean}`

```latex
Equation \eqref{pythagorean} is called “勾股定理” in Chinese.
```

具体公式用axmath即可，或者用ai识别。

### 字体字号

一般不用管（你猜我为啥用LaTeX）。

### 段落格式与间距

知道这玩意能改就行（你猜我为啥不用word）。

### 段落格式

同上（真要改问ai就行）。

### 水平/垂直间距

一样。

### 页面/分栏

如上。

### 页眉/页脚

命令`\pagestyle`来修改页眉页脚的样式:

```latex
\pagestyle{〈page-style〉}
```

命令`\thispagestyle`只影响当页的页眉页脚样式
〈page-style〉参数为样式的名称，预定义了四类样式。其中headings的情况较为复杂:

```latex
empty
页眉页脚为空
plain
页眉为空, 页脚为页码。(article和report文档类默认; book文档类的每章第一页也为plain格式)
headings
页眉为章节标题和页码，页脚为空。(book文档类默认)
myheadings
页眉为页码及\markboth和\markright命令手动指定的内容,页脚为空。
```

### 参考文献——BIBTEX

BIBTEX是最为流行的参考文献数据组织格式之一。它的出现让我们摆脱手写参考文献条目的麻烦。我们还可以通过参考文献样式的支持，让同一份BIBTEX数据库生成不同样式的参考文献列表。
BIBTEX数据库以.bib作为扩展名，其内容是若干个文献条目，每个条目的格式为：

```latex
@〈type〉{〈citation〉,
    〈key1〉={〈value9〉},
    〈key2〉={〈value2〉},
    …
}
```

其中(type)为文献的类别,如article为学术论文, book为书籍, incollection为论文集中的某一篇,等等。〈citation〉为`\cite`命令使用的文献标签。在`〈citation〉`之后为条目里的各个字段,以`〈key〉={〈value〉}`的形式组织。

一般来说，不需要自己写，在出处复制即可。

在同一目录下创一个文本文件，粘贴进去，后缀改成.bib，在导言区指定引用格式：

```latex
\bibliographystyle{plain}
```

引用时，

```latex
\cite{一串数字}
```

在文章最后（列出参考文献时）：

```latex
\bibliography{你的.bib文件名}
```

