package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class Weather {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private long dataNo; // 필드명을 CamelCase로 변경
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	private LocalDateTime time;
	private int stn;
	
	private double wd1;
	private double wd2;
	private double wds;
	private double wss;
	private double wd10;
	private double ws10;
	private double ta;
	private double re;
	private double rn15m;
	private double rn60m;
	private double rn12h;
	private double rnday;
	private double hm;
	private double pa;
	private double ps;
	private double td; 
}
