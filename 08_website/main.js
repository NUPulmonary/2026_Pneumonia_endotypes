import { Legend } from "./d3-color-legend.js";

function formatNumber(n) {
    let s = '<span class="number">';
    if (n < 0) {
        s += "–" + (-n).toString();
    } else {
        s += n.toString();
    }
    if (/e(-)?\d+$/.test(n)) {
        const parts = n.split('e', 2);
        s = '<span class="number">' + parts[0] + `&hairsp;<i>&times;</i>&hairsp;10<sup>${parts[1].replace('-', '–')}</sup>`;
    }
    return s + "</span>";
}

function displayCovariate(txt) {
    if (txt.indexOf("$") === -1) {
        return txt;
    }
    while (txt.indexOf("$") !== -1) {
        const start = txt.indexOf("$");
        const end = txt.indexOf("$", start + 1);
        if (end === -1) break;
        let mathExpr = txt.substring(start + 1, end);
        if (mathExpr[0] === "_") {
            mathExpr = mathExpr.substring(1);
            if (mathExpr[0] === "{" && mathExpr[mathExpr.length - 1] === "}") {
                mathExpr = mathExpr.substring(1, mathExpr.length - 1);
                if (mathExpr.startsWith("\\mathrm")) {
                    mathExpr = mathExpr.substring(8, mathExpr.length - 1);
                }
            }
            mathExpr = `<tspan dy="4" font-size="10">${mathExpr}</tspan><tspan dy="-4">&#8203;</tspan>`;
        } else if (mathExpr[0] === "^") {
            mathExpr = mathExpr.substring(1);
            mathExpr = mathExpr.replace("\\dag", "&dagger;");
            mathExpr = `<tspan dy="-4" font-size="10">${mathExpr}</tspan><tspan dy="4">&#8203;</tspan>`;
        }

        txt = txt.substring(0, start) + mathExpr + txt.substring(end + 1);
    }
    return txt;
}

// Function to determine if a color is light (needs dark text) or dark (needs light text)
const isLightColor = (color) => {
    const rgb = d3.rgb(color);
    // Calculate perceived brightness
    const brightness = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    return brightness > 0.65;
};

// Function to map covariate names between display (heatmap) and CSV columns
function mapCovariateName(displayName, direction = "toCSV") {
    // Mapping from display name to CSV column name
    const nameMap = {
        // Add specific mappings here if display names differ from CSV names
        // e.g., 'Age Group': 'age_group',
        "VAP cure in 7 days": "VAP_is_cured_d7",
        "Episode cured": "Episode_is_cured",
        ECMO: "ECMO_flag",
        Immunocompromised: "Immunocompromised_flag",
        Tracheostomy: "Tracheostomy_flag",
        CRRT: "CRRT_flag",
        Hemodialysis: "Hemodialysis_flag",
        Norepinephrine: "Norepinephrine_flag",
        "Virus not detected": "Pathogen_virus_detected",
        "Bacteria detected": "Pathogen_bacteria_detected",
        "Female sex": "Sex",
        Mortality: "Binary_outcome",
        "CAP / HAP / VAP / NPC": "Episode_category",
        "Healthy / NPC / Viral, bacterial or mixed": "Healthy_vs_NPC_vs_Pathogen",
        "Flow cytometry clusters": "Flow_clusters",
        "scRNA-seq clusters": "ScRNAseq_clusters",
        "Fungi detected": "Pathogen_fungi_detected",
        "Adjudicated pathogen": "Episode_etiology",
        "Overall adjudicated outcome": "Global_cause_failure",
        "Days on ventilator": "days_on_ventilator",
        "P$_{\\mathrm{a}}$O$_2$/F$_{\\mathrm{i}}$O$_2$": "PaO2FIO2_ratio",
        "P$_{\\mathrm{a}}$CO$_2$": "ABG_PaCO2",
        "Minute ventilation": "Minute_Ventilation",
        "D-dimer": "D_dimer_ff",
        "BAL neutrophils (%)": "BAL_pct_neutrophils",
        "BAL lymphocytes (%)": "BAL_pct_lymphocytes",
        "ICU antibiotic days to date": "days_of_icu_abx_until_today",
        "ICU steroid dose to date": "cumulative_icu_steroid_dose_until_today",
        "No smoking history": "Smoking_status",
    };

    if (direction === "toCSV") {
        // Return CSV column name
        let name = nameMap[displayName] || displayName;
        return name.replaceAll(" ", "_");
    } else {
        // Return display name from CSV column name
        const reverseMap = Object.fromEntries(
            Object.entries(nameMap).map(([display, csv]) => [csv, display]),
        );
        let name = reverseMap[displayName] || displayName;
        return name.replaceAll(" ", "_");
    }
}

function getCovariateNameForGraph(covariate) {
    const map = {
        "Virus not detected": "Virus detected",
        "Bacteria detected$^\\dag$": "Bacteria detected",
        "CAP / HAP / VAP / NPC": "Pneumonia episode category",
        "Healthy / NPC / Any pathogen": "Infection status",
        "No smoking history": "Smoking history",
        "Female sex": "Sex",
    };
    if (map[covariate] === undefined) {
        return covariate;
    }
    return map[covariate];
}

// Function to calculate boxplot statistics
function calculateBoxplotStats(values) {
    const sorted = values.slice().sort(d3.ascending);
    const q1 = d3.quantile(sorted, 0.25);
    const median = d3.quantile(sorted, 0.5);
    const q3 = d3.quantile(sorted, 0.75);
    const iqr = q3 - q1;
    const min = d3.max([d3.min(sorted), q1 - 1.5 * iqr]);
    const max = d3.min([d3.max(sorted), q3 + 1.5 * iqr]);
    return { min, q1, median, q3, max, iqr };
}

function setupGraph(element, dimensions, xScale, xLabel, yScale, yLabel, title, options) {
    options ||= {};

    element.html("");

    const svg = element
        .append("svg")
        .attr("width", dimensions.width + dimensions.left + dimensions.right)
        .attr("height", dimensions.height + dimensions.top + dimensions.bottom)
        .append("g")
        .attr("transform", `translate(${dimensions.left}, 0)`);

    const header = d3
        .select(svg.node().parentNode)
        .append("g")
        .attr("transform", `translate(${dimensions.left}, 0)`);

    // Add title
    header.append("text").attr("x", 0).attr("y", 16).style("font-size", "16px").html(title);

    // Add X axis
    svg.append("g")
        .classed("graph-x-axis", true)
        .attr("transform", `translate(0, ${dimensions.height})`)
        .call(d3.axisBottom(xScale))
        .style("font-size", "12px");

    if (options.modifyXAxis !== undefined) {
        options.modifyXAxis(svg.select(".graph-x-axis"));
    }
    const xAxisHeight = svg.select(".graph-x-axis").node().getBBox().height;
    // Add X axis label
    svg.append("text")
        .attr("x", dimensions.width / 2)
        .attr("y", dimensions.height + xAxisHeight + 16)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .html(xLabel);

    // Add Y axis
    svg.append("g")
        .classed("graph-y-axis", true)
        .call(d3.axisLeft(yScale))
        .style("font-size", "12px");

    const yAxisWidth = svg.select(".graph-y-axis").node().getBBox().width;

    // Add Y axis label
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -yAxisWidth - 8)
        .attr("x", -dimensions.height / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .html(yLabel);

    const headerHeight = header.node().getBBox().height;
    svg.attr("transform", `translate(${dimensions.left}, ${headerHeight + 12})`);

    return {
        graph: svg,
        header: header,
    };
}

function displayHeatmap(data, element, dimensions, options) {
    if (!Array.isArray(data[0])) {
        data = [data, data];
    }
    options ||= {};
    options.displayCovariate ||= displayCovariate;
    const colorData = data[0];
    const pvalData = data[1];
    const width = dimensions.width - dimensions.left - dimensions.right;
    const height = dimensions.height - dimensions.top - dimensions.bottom;
    // Create SVG
    const svg = element
        .append("svg")
        .attr("width", dimensions.width)
        .attr("height", dimensions.height)
        .append("g")
        .attr("transform", `translate(${dimensions.left}, ${dimensions.top})`);

    let factors = [];
    const covariates = [];
    // Create a matrix for the heatmap data
    const matrix = colorData.map((item, idx) => {
        const pvalItem = pvalData[idx];
        const row = { Covariate: item[""] };
        covariates.push(row.Covariate);

        Object.keys(item).forEach((key) => {
            if (key === "") return;
            row[key] = {
                colorValue: +item[key],
                pval: pvalItem ? +pvalItem[key] : 1,
            };
            if (!factors.includes(key)) {
                factors.push(key);
            }
        });
        return row;
    });
    factors = options.colnames || factors;

    // Create scales
    let paddingProp = (factors.length - 1) / width;
    const x = d3.scaleBand().domain(factors).range([0, width]).paddingInner(paddingProp);

    paddingProp = (covariates.length - 1) / height;
    const y = d3.scaleBand().domain(covariates).range([0, height]).paddingInner(paddingProp);

    const allColorValues = matrix.flatMap((d) => factors.map((factor) => d[factor].colorValue));
    const extent = d3.extent(allColorValues);
    const colorScale = options.colorScale(extent);

    // Add row labels (terms)
    svg.append("g")
        .selectAll("text")
        .data(covariates)
        .enter()
        .append("text")
        .attr("class", "row-label")
        .attr("x", -10)
        .attr("y", (d) => y(d) + y.bandwidth() / 2)
        .attr("dy", ".32em")
        .html((d) => options.displayCovariate(d));

    svg.append("g")
        .selectAll("text")
        .data(covariates.slice(1))
        .enter()
        .append("path")
        .attr(
            "d",
            (d) =>
                `M ${x(factors[0])} ${y(d) - 0.5} H ${x(factors[factors.length - 1]) + x.bandwidth()}`,
        )
        .attr("stroke", "#d1d5df")
        .attr("stroke-width", 0.5);

    // Add column labels at the bottom
    const columnLabels = svg
        .append("g")
        .attr("transform", `translate(0, ${height + 15})`)
        .selectAll("text")
        .data(factors)
        .enter()
        .append("text")
        .attr("class", "column-label")
        .attr("x", 0)
        .attr("y", 0)
        .attr("text-anchor", "end")
        .attr("transform", (d) => `translate(${x(d) + x.bandwidth() / 2 + 4}, 0) rotate(-30)`)
        // .attr('transform-origin', 'bottom right')
        .text((d, i) => (options.colnames && options.colnames[i]) || `Factor ${i + 1}`);
    if (options.columnClick) {
        columnLabels
            .on("click", (event, d) => {
                options.columnClick(event, d, factors.indexOf(d));
            })
            .classed("as-link", true);
    }

    svg.append("g")
        .selectAll("text")
        .data(factors.slice(1))
        .enter()
        .append("path")
        .attr(
            "d",
            (d) =>
                `M ${x(d) - 0.5} ${y(covariates[0])} V ${y(covariates[covariates.length - 1]) + y.bandwidth()}`,
        )
        .attr("stroke", "#d1d5df")
        .attr("stroke-width", 0.5);

    // Create heatmap cells
    const cells = svg
        .append("g")
        .selectAll("rect")
        .data(
            matrix.flatMap((d) =>
                factors.map((factor) => ({
                    covariate: d.Covariate,
                    factor: factor,
                    factorIndex: factors.indexOf(factor),
                    value: d[factor],
                })),
            ),
        )
        .enter();

    // Add cell rectangles
    cells
        .append("rect")
        .attr("class", (d) => (d.value && d.value !== "" ? "cell" : "cell cell-na"))
        .attr("x", (d) => x(d.factor))
        .attr("y", (d) => y(d.covariate))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("fill", (d) => {
            return colorScale(d.value.colorValue);
        })
        .on("mouseover", function (event, d) {
            // Only show tooltip for cells with values
            if (d.value !== undefined && d.value !== "") {
                options.mouseover && options.mouseover(event, d);
                d3.select(this).classed("hover", true).raise();
            }
        })
        .on("mouseout", function (event, d) {
            options.mouseout && options.mouseout(event, d);
            d3.select(this).classed("hover", false);
        })
        .on("click", function (event, d) {
            if (d.value !== undefined && d.value !== "") {
                event.stopPropagation();

                // Remove highlight from all cells
                d3.selectAll("rect.cell.highlighted").classed("highlighted", false);

                // Highlight selected cell
                d3.select(this).classed("highlighted", true);

                options.click(event, d);
            }
        });

    // Add star to significant cells (padj < 0.05)
    const cellSignif = svg
        .append("g")
        .selectAll("text")
        .data(
            matrix.flatMap((d) =>
                factors.map((factor) => ({
                    covariate: d.Covariate,
                    factor: factor,
                    factorIndex: factors.indexOf(factor),
                    value: d[factor],
                })),
            ),
        )
        .enter();
    let SIGNIF_THRESHOLD = 0.05;
    if (options.logPval !== false) {
        SIGNIF_THRESHOLD = -Math.log10(SIGNIF_THRESHOLD);
    }
    cellSignif
        .filter(
            (d) =>
                d.value &&
                d.value !== "" &&
                ((options.logPval !== false && d.value.pval > SIGNIF_THRESHOLD) ||
                    (options.logPval === false && d.value.pval < SIGNIF_THRESHOLD)),
        )
        .append("text")
        .attr("class", "cell-text")
        .attr("x", (d) => x(d.factor) + x.bandwidth() / 2)
        .attr("y", (d) => y(d.covariate) + y.bandwidth() / 2 + 6)
        .text("*") // Add star for significant values
        .style("fill", (d) => {
            const cellColor = colorScale(d.value.colorValue);
            return isLightColor(cellColor) ? "black" : "white";
        })
        .style("font-size", "20px");

    const legendWidth = 200;
    const legend =
        (options.legend && options.legend(extent, legendWidth)) ||
        Legend(colorScale, {
            title: "–log10(padj)",
            width: legendWidth,
        });
    let cat = element.node();
    cat.parentNode.insertBefore(legend, cat.nextSibling);
    d3.select(legend)
        .attr("transform", "rotate(-90)")
        .attr("transform-origin", "top left")
        .attr("height", legendWidth)
        .attr("width", 50)
        .attr("viewBox", `0,0,50,${legendWidth}`)
        .style("margin-top", `${legendWidth}px`)
        .style("margin-left", `-${dimensions.right - 10}px`);
}

function processHallmarkName(name) {
    const SHORT_WORDS = ["ACID", "VIA", "BETA", "LATE", "BILE", "HEME"];
    let result = [];
    let words = name.split("_");
    words.forEach((word) => {
        if (/\d/.test(word) || (word.length < 5 && SHORT_WORDS.indexOf(word) === -1)) {
            result.push(word);
        } else {
            result.push(word.toLowerCase());
        }
    });
    result = result.join(" ");
    result = result.replace("TNFA", "TNFα").replace("NFKB", "NFκB").replace("TGF beta", "TGFβ").replace("WNT beta ", "WNT/β-");
    result = result[0].toUpperCase() + result.substr(1);
    return result;
}

// Function to format gene list
function formatGeneList(geneString) {
    if (!geneString || geneString.trim() === "") {
        return "No genes in leading edge";
    }

    // Format the gene list
    // Assuming genes are separated by some delimiter like comma or space
    const genes = geneString.split(/;/).filter((gene) => gene.trim() !== "");

    if (genes.length === 0) {
        return "No genes in leading edge";
    }

    return genes.join(", ");
}

function fetchCsvGz(path) {
    return d3.blob(path).then((blob) => {
        const ds = new DecompressionStream("gzip");
        const decompressedStream = blob.stream().pipeThrough(ds);
        return new Response(decompressedStream).text().then((data) => {
            return d3.csvParse(data);
        });
    });
}

const GSEA_LIST = [
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_IL2_STAT5_SIGNALING",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_COAGULATION",
    "HALLMARK_ALLOGRAFT_REJECTION",
    "HALLMARK_PROTEIN_SECRETION",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MITOTIC_SPINDLE",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_APOPTOSIS",

    "HALLMARK_HYPOXIA",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_FATTY_ACID_METABOLISM",
    "HALLMARK_PEROXISOME",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "HALLMARK_XENOBIOTIC_METABOLISM",

    "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",

    "HALLMARK_DNA_REPAIR",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_NOTCH_SIGNALING",
];

const MOFA_ROTATIONS = [
    "Pathogen groups: Early SARS-CoV-2",
    "Bacteria only detected",
    "Bacteria detected",
    "Immunocompromised",
    "Female sex",
    "Mortality",
    "ECMO",
    "Episode category: CAP",
    "VAP cure in 7 days",
    "Viral, bacterial or mixed",
];

const CELL_TYPE_ORDER = [
    "NUPR1+ AM",
    "MRC1+C1QA+ AM",
    "MRC1+C1QA– AM",
    "DC2",
    "Classical monocytes-2 IL1B",
    "Classical monocytes-1 CCR2",
    "Non-classical monocytes",
    "Interstitial macrophages",
    "Proliferating NUPR1+ AM",

    "CD4 T cells",
    "CD8 T cells",
    "γδT cells",
    "Tregs",
    "Proliferating CD4 T cells",
    "Proliferating CD8 T cells",

    "B cells",
    "Plasma cells",
    "Proliferating plasma cells",

    "Ciliated cells",
    "Secretory cells",
];

export {
    displayCovariate,
    isLightColor,
    displayHeatmap,
    mapCovariateName,
    getCovariateNameForGraph,
    setupGraph,
    calculateBoxplotStats,
    Legend,
    formatNumber,
    processHallmarkName,
    formatGeneList,
    GSEA_LIST,
    MOFA_ROTATIONS,
    CELL_TYPE_ORDER,
    fetchCsvGz,
};
